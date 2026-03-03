# ============================================================================
# PETROQUANT — MODULAR STRATEGY FRAMEWORK
# ============================================================================
# Contains:
#   BaseStrategy       — abstract base class for all strategies
#   HMMXGBoostStrategy — HMM regime detection + XGBoost walk-forward
#   TD0RLStrategy      — Reinforcement learning with TD(0) temporal difference
#
# USAGE:
#   from strategy import HMMXGBoostStrategy, TD0RLStrategy
#   strat = HMMXGBoostStrategy(fwd_days=5)
#   result_df = strat.run(master_df)
#
# TO ADD A NEW STRATEGY:
#   1. Subclass BaseStrategy
#   2. Implement engineer_features(), fit_predict(), forecast_returns()
#   3. Import & run in run_strategy.py
# ============================================================================

import numpy as np
import pandas as pd
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from hmmlearn.hmm import GaussianHMM
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')


# ═════════════════════════════════════════════════════════════════════════════
# BASE STRATEGY
# ═════════════════════════════════════════════════════════════════════════════
class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    def __init__(self, name="BaseStrategy"):
        self.name = name
        self.metadata = {}

    @abstractmethod
    def engineer_features(self, master_df):
        """Transform raw master_df into engineered features. Must be backward-looking only."""
        pass

    @abstractmethod
    def fit_predict(self, feat_df):
        """Train model(s) and generate Signal, Probability, Prediction columns."""
        pass

    @abstractmethod
    def forecast_returns(self, feat_df):
        """Generate multi-horizon return forecasts."""
        pass

    def run(self, master_df):
        """Full pipeline: features → fit/predict → forecast → return enriched df."""
        print(f"\n{'='*70}")
        print(f"  RUNNING STRATEGY: {self.name}")
        print(f"{'='*70}\n")

        feat_df = self.engineer_features(master_df)
        feat_df = self.fit_predict(feat_df)
        feat_df = self.forecast_returns(feat_df)

        print(f"\n{'='*70}")
        print(f"  ✓ {self.name} COMPLETE — {len(feat_df)} rows with signals")
        print(f"{'='*70}\n")
        return feat_df

    def get_metadata(self):
        """Return strategy metadata for the dashboard."""
        return self.metadata


# ═════════════════════════════════════════════════════════════════════════════
# HMM + XGBOOST REGIME-AWARE STRATEGY
# ═════════════════════════════════════════════════════════════════════════════
class HMMXGBoostStrategy(BaseStrategy):
    """
    3-state HMM regime detection + walk-forward XGBoost classifier.
    Generates directional signals and multi-horizon return forecasts.
    """

    def __init__(self, fwd_days=5, hmm_states=3, initial_train_days=500,
                 retrain_every=63, buy_threshold=0.55, sell_threshold=0.45):
        super().__init__(name="HMM + XGBoost Regime-Aware")
        self.fwd_days = fwd_days
        self.hmm_states = hmm_states
        self.initial_train_days = initial_train_days
        self.retrain_every = retrain_every
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

        # Stored after fitting
        self.regime_map = {}
        self.regime_stats = None
        self.fold_metrics = []
        self.feature_importance = None
        self.feature_cols = []
        self.forecast_horizons = [1, 7, 15, 30, 60, 90, 180]

    # ── HELPERS ──────────────────────────────────────────────────────────
    @staticmethod
    def _resolve_col(df, preferred, fallbacks):
        for name in [preferred] + fallbacks:
            if name in df.columns:
                return name
        return preferred

    # ── STEP 1: FEATURE ENGINEERING ──────────────────────────────────────
    def engineer_features(self, master_df):
        print("  [1/4] Engineering features...")
        feat = master_df.copy()

        COL_CRUDE = self._resolve_col(feat, 'Crude_Stocks_1000bbl', ['Crude_Stocks_1000bb1'])
        COL_SPR   = self._resolve_col(feat, 'SPR_Stocks_1000bbl', ['SPR_Stocks_1000bb1'])
        self.metadata['col_crude'] = COL_CRUDE
        self.metadata['col_spr'] = COL_SPR

        # 1. Price-Derived
        feat['WTI_LogRet_1d']   = np.log(feat['WTI_Close'] / feat['WTI_Close'].shift(1))
        feat['WTI_LogRet_5d']   = np.log(feat['WTI_Close'] / feat['WTI_Close'].shift(5))
        feat['Brent_LogRet_1d'] = np.log(feat['Brent_Close'] / feat['Brent_Close'].shift(1))
        feat['Brent_LogRet_5d'] = np.log(feat['Brent_Close'] / feat['Brent_Close'].shift(5))
        feat['Spread']          = feat['WTI_Close'] - feat['Brent_Close']
        spread_mu  = feat['Spread'].rolling(20).mean()
        spread_sig = feat['Spread'].rolling(20).std()
        feat['Spread_Zscore'] = (feat['Spread'] - spread_mu) / (spread_sig + 1e-8)
        feat['WTI_Mom_10'] = feat['WTI_Close'].pct_change(10)
        feat['WTI_Mom_20'] = feat['WTI_Close'].pct_change(20)

        # 2. Volatility-Derived
        if 'OVX' in feat.columns:
            feat['OVX_Chg_1d'] = feat['OVX'].pct_change(1)
            feat['OVX_Chg_5d'] = feat['OVX'].pct_change(5)
        feat['RealizedVol_20d'] = feat['WTI_LogRet_1d'].rolling(20).std() * np.sqrt(252)

        # 3. Macro-Derived
        if 'USD_Index' in feat.columns:
            feat['USD_Chg_1d']  = feat['USD_Index'].pct_change(1)
            feat['USD_Chg_5d']  = feat['USD_Index'].pct_change(5)
            feat['USD_ROC_10']  = feat['USD_Index'].pct_change(10)

        # 4. Positioning-Derived
        if 'Net_Speculative_Position' in feat.columns:
            feat['NSP_Chg_5d'] = feat['Net_Speculative_Position'].pct_change(5)
            nsp_mu  = feat['Net_Speculative_Position'].rolling(20).mean()
            nsp_sig = feat['Net_Speculative_Position'].rolling(20).std()
            feat['NSP_Zscore'] = (feat['Net_Speculative_Position'] - nsp_mu) / (nsp_sig + 1e-8)

        # 5. Supply-Derived
        if COL_CRUDE in feat.columns:
            feat['Crude_Chg_5d'] = feat[COL_CRUDE].pct_change(5)
        if 'US_Oil_Rigs' in feat.columns:
            feat['Rigs_Chg_5d'] = feat['US_Oil_Rigs'].pct_change(5)
        if COL_SPR in feat.columns:
            feat['SPR_Chg_5d'] = feat[COL_SPR].pct_change(5)

        # 6. Crack Spread
        if 'Crack_3_2_1' in feat.columns:
            crack_mu  = feat['Crack_3_2_1'].rolling(20).mean()
            crack_sig = feat['Crack_3_2_1'].rolling(20).std()
            feat['Crack_Zscore'] = (feat['Crack_3_2_1'] - crack_mu) / (crack_sig + 1e-8)
            feat['Crack_Chg_5d'] = feat['Crack_3_2_1'].pct_change(5)

        # Clean up
        feat = feat.replace([np.inf, -np.inf], np.nan).dropna()

        # Drop zero-variance
        zero_var = [c for c in feat.columns if feat[c].std() == 0]
        if zero_var:
            print(f"    Dropping zero-variance: {zero_var}")
            feat = feat.drop(columns=zero_var)

        # Identify raw vs engineered columns
        raw_cols = ['WTI_Close', 'Brent_Close', 'OVX', 'USD_Index', 'Crack_3_2_1',
                    'Net_Speculative_Position', COL_CRUDE, 'US_Oil_Rigs', COL_SPR, 'Spread']
        engineered = [c for c in feat.columns if c not in raw_cols]
        print(f"    ✓ {len(engineered)} engineered features from {len(raw_cols)} raw variables")
        print(f"    ✓ Clean DataFrame: {feat.shape[0]} rows × {feat.shape[1]} cols")

        return feat

    # ── STEP 2+3: HMM + XGBOOST FIT/PREDICT ─────────────────────────────
    def fit_predict(self, feat_df):
        feat = feat_df.copy()

        # ── HMM Regime Detection ─────────────────────────────────────────
        print("  [2/4] Fitting HMM regime detection...")
        X_hmm = feat[['WTI_LogRet_1d', 'RealizedVol_20d']].values
        scaler_hmm = StandardScaler()
        X_hmm_scaled = scaler_hmm.fit_transform(X_hmm)

        hmm_model = GaussianHMM(
            n_components=self.hmm_states, covariance_type='full',
            n_iter=200, random_state=42
        )
        hmm_model.fit(X_hmm_scaled)
        feat['HMM_State'] = hmm_model.predict(X_hmm_scaled)

        # Label regimes by volatility + return characteristics
        state_stats = pd.DataFrame({
            'State': range(self.hmm_states),
            'Mean_Return': [feat.loc[feat['HMM_State'] == s, 'WTI_LogRet_1d'].mean()
                            for s in range(self.hmm_states)],
            'Mean_Vol': [feat.loc[feat['HMM_State'] == s, 'RealizedVol_20d'].mean()
                         for s in range(self.hmm_states)],
            'Count': [int((feat['HMM_State'] == s).sum()) for s in range(self.hmm_states)]
        })
        state_stats = state_stats.sort_values('Mean_Vol', ascending=False)
        states_ordered = state_stats['State'].tolist()

        regime_map = {}
        regime_map[states_ordered[0]] = 'PANIC'
        remaining = states_ordered[1:]
        if state_stats.loc[state_stats['State'] == remaining[0], 'Mean_Return'].values[0] > \
           state_stats.loc[state_stats['State'] == remaining[1], 'Mean_Return'].values[0]:
            regime_map[remaining[0]] = 'BULL'
            regime_map[remaining[1]] = 'CHOPPY'
        else:
            regime_map[remaining[1]] = 'BULL'
            regime_map[remaining[0]] = 'CHOPPY'

        feat['Regime'] = feat['HMM_State'].map(regime_map)
        self.regime_map = regime_map
        self.regime_stats = state_stats

        print(f"    ✓ HMM converged — Log-likelihood: {hmm_model.score(X_hmm_scaled):.2f}")
        for _, row in state_stats.iterrows():
            s = int(row['State'])
            name = regime_map[s]
            pct = row['Count'] / len(feat) * 100
            print(f"      State {s} → {name:7s} | ret={row['Mean_Return']:+.5f} "
                  f"vol={row['Mean_Vol']:.4f} | {int(row['Count'])} days ({pct:.1f}%)")

        # ── Target ───────────────────────────────────────────────────────
        fwd_ret = feat['WTI_Close'].shift(-self.fwd_days) / feat['WTI_Close'] - 1
        feat['Fwd_Return'] = fwd_ret
        feat['Target'] = (fwd_ret > 0).astype(int)
        feat = feat.dropna(subset=['Target'])

        # ── Feature selection ────────────────────────────────────────────
        exclude_cols = [
            'WTI_Close', 'Brent_Close', 'OVX', 'USD_Index', 'Crack_3_2_1',
            'Net_Speculative_Position', 'Crude_Stocks_1000bbl', 'Crude_Stocks_1000bb1',
            'US_Oil_Rigs', 'SPR_Stocks_1000bbl', 'SPR_Stocks_1000bb1',
            'Cushing_Stocks_1000bbl', 'Cushing_Stocks_1000bb1', 'Spare_Capacity_mbpd',
            'Spread', 'Fwd_Return', 'Target', 'Probability',
            'Prediction', 'Signal', 'Signal_Label', 'Regime', 'HMM_State'
        ]
        self.feature_cols = [c for c in feat.columns if c not in exclude_cols]

        # ── Walk-Forward XGBoost ─────────────────────────────────────────
        print(f"  [3/4] Walk-forward XGBoost ({len(self.feature_cols)} features, "
              f"{self.fwd_days}-day target)...")

        X = feat[self.feature_cols].values
        y = feat['Target'].values
        dates = feat.index
        n = len(X)

        predictions  = np.full(n, np.nan)
        probabilities = np.full(n, np.nan)
        scaler = StandardScaler()
        self.fold_metrics = []

        i = self.initial_train_days
        while i < n:
            end = min(i + self.retrain_every, n)
            X_train, y_train = X[:i], y[:i]
            X_test = X[i:end]

            scaler.fit(X_train)
            X_train_s = scaler.transform(X_train)
            X_test_s  = scaler.transform(X_test)

            xgb_model = XGBClassifier(
                max_depth=4, n_estimators=300, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                objective='binary:logistic', eval_metric='logloss',
                use_label_encoder=False, random_state=42, verbosity=0
            )
            xgb_model.fit(X_train_s, y_train)

            probs = xgb_model.predict_proba(X_test_s)[:, 1]
            preds = (probs > 0.5).astype(int)
            probabilities[i:end] = probs
            predictions[i:end]   = preds

            acc = accuracy_score(y[i:end], preds)
            self.fold_metrics.append({
                'fold': len(self.fold_metrics) + 1,
                'train': i, 'test': end - i,
                'period': f"{dates[i].date()} → {dates[end-1].date()}",
                'acc': acc
            })
            i = end

        feat['Probability'] = probabilities
        feat['Prediction']  = predictions

        # Feature importance from last model
        self.feature_importance = pd.Series(
            xgb_model.feature_importances_, index=self.feature_cols
        ).sort_values(ascending=False)

        # ── Signals ──────────────────────────────────────────────────────
        mask = feat['Probability'].notna()
        feat['Signal'] = 0
        feat.loc[mask & (feat['Probability'] > self.buy_threshold), 'Signal'] = 1
        feat.loc[mask & (feat['Probability'] < self.sell_threshold), 'Signal'] = -1
        feat['Signal_Label'] = feat['Signal'].map({1: 'BUY', -1: 'SELL', 0: 'HOLD'})

        fold_df = pd.DataFrame(self.fold_metrics)
        avg_acc = fold_df['acc'].mean()
        print(f"    ✓ {len(fold_df)} folds | Avg OOS accuracy: {avg_acc:.1%}")

        oos = feat[mask]
        n_buy  = (oos['Signal'] == 1).sum()
        n_sell = (oos['Signal'] == -1).sum()
        n_hold = (oos['Signal'] == 0).sum()
        total  = len(oos)
        print(f"    ✓ Signals: BUY={n_buy} ({n_buy/total*100:.1f}%) | "
              f"SELL={n_sell} ({n_sell/total*100:.1f}%) | "
              f"HOLD={n_hold} ({n_hold/total*100:.1f}%)")

        self.metadata.update({
            'avg_accuracy': avg_acc,
            'n_folds': len(fold_df),
            'n_features': len(self.feature_cols),
            'fwd_days': self.fwd_days,
            'has_regimes': True,
        })

        return feat

    # ── STEP 4: MULTI-HORIZON FORECASTING ────────────────────────────────
    def forecast_returns(self, feat_df):
        print(f"  [4/4] Multi-horizon return forecasting ({self.forecast_horizons})...")
        feat = feat_df.copy()

        if not self.feature_cols:
            print("    ⚠ No feature columns — skipping forecasts")
            return feat

        X_all = feat[self.feature_cols].values
        scaler = StandardScaler()
        forecasts = {}

        for horizon in self.forecast_horizons:
            # Build target: forward return for this horizon
            fwd = feat['WTI_Close'].shift(-horizon) / feat['WTI_Close'] - 1
            valid_mask = fwd.notna()
            valid_idx = feat.index[valid_mask]

            if valid_mask.sum() < self.initial_train_days + 50:
                print(f"    ⚠ {horizon}d — not enough data, skipping")
                continue

            X_h = X_all[valid_mask]
            y_dir = (fwd[valid_mask] > 0).astype(int).values
            y_ret = fwd[valid_mask].values

            # Use last 80% for training, predict on the very last observation
            train_n = int(len(X_h) * 0.8)
            if train_n < 100:
                continue

            scaler.fit(X_h[:train_n])
            X_train_s = scaler.transform(X_h[:train_n])
            X_last_s  = scaler.transform(X_h[-1:])

            # Direction classifier
            clf = XGBClassifier(
                max_depth=4, n_estimators=200, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                objective='binary:logistic', eval_metric='logloss',
                use_label_encoder=False, random_state=42, verbosity=0
            )
            clf.fit(X_train_s, y_dir[:train_n])
            prob_up = clf.predict_proba(X_last_s)[0, 1]

            # Magnitude regressor
            reg = XGBRegressor(
                max_depth=4, n_estimators=200, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                objective='reg:squarederror', random_state=42, verbosity=0
            )
            reg.fit(X_train_s, y_ret[:train_n])
            exp_return = float(reg.predict(X_last_s)[0])

            forecasts[horizon] = {
                'prob_up': float(prob_up),
                'expected_return': exp_return,
            }
            direction = "▲" if prob_up > 0.5 else "▼"
            print(f"    ✓ {horizon:3d}d → P(up)={prob_up:.1%} | "
                  f"E[ret]={exp_return:+.2%} {direction}")

        self.metadata['forecasts'] = forecasts
        return feat


# ═════════════════════════════════════════════════════════════════════════════
# TD(0) REINFORCEMENT LEARNING STRATEGY — SEMI-GRADIENT WITH LINEAR FA
# ═════════════════════════════════════════════════════════════════════════════
class TD0RLStrategy(BaseStrategy):
    """
    Advanced trading strategy using Semi-Gradient TD(0) with Linear
    Function Approximation.

    Key improvements over basic tabular TD:
    - Linear Q(s,a) = w_a · φ(s) instead of lookup table
    - 12 continuous state features (no crude binning)
    - Differential Sharpe Ratio reward (Moody & Saffell 1998)
    - Softmax/Boltzmann exploration with temperature decay
    - Risk-sensitive reward with drawdown penalty
    - 5 training epochs on initial window
    - Walk-forward online learning
    """

    ACTIONS = [1, 0, -1]  # BUY, HOLD, SELL

    def __init__(self, alpha=0.005, gamma=0.99, tau_start=1.0,
                 tau_end=0.1, transaction_cost=0.0005,
                 initial_train_days=400, retrain_every=63,
                 n_epochs=5, drawdown_penalty=2.0, eta=0.01):
        super().__init__(name="TD(0) Reinforcement Learning")
        self.alpha = alpha                  # learning rate
        self.gamma = gamma                  # discount factor
        self.tau_start = tau_start          # softmax temperature start
        self.tau_end = tau_end              # softmax temperature end
        self.transaction_cost = transaction_cost
        self.initial_train_days = initial_train_days
        self.retrain_every = retrain_every
        self.n_epochs = n_epochs            # training epochs on initial data
        self.drawdown_penalty = drawdown_penalty
        self.eta = eta                      # differential Sharpe EMA decay

        # Weight vectors: one per action → w[action] is a numpy array
        self.n_features = 0
        self.weights = {}                   # action → weight vector
        self.feature_names = []
        self.feature_importance = None
        self.feature_cols = []
        self.fold_metrics = []
        self.forecast_horizons = [1, 7, 15, 30, 60, 90, 180]

        # Running stats for differential Sharpe
        self._A = 0.0   # EMA of returns
        self._B = 0.0   # EMA of squared returns

    # ── STATE FEATURE VECTOR ─────────────────────────────────────────────
    def _build_state_features(self, feat_df):
        """
        Build continuous state feature matrix from the DataFrame.
        All features are normalized to roughly [-1, 1] using rolling z-scores.
        """
        sf = pd.DataFrame(index=feat_df.index)

        # 1. Multi-timeframe momentum
        sf['mom_5d']  = feat_df['WTI_Close'].pct_change(5)
        sf['mom_10d'] = feat_df['WTI_Close'].pct_change(10)
        sf['mom_20d'] = feat_df['WTI_Close'].pct_change(20)
        sf['mom_60d'] = feat_df['WTI_Close'].pct_change(60)

        # 2. Trend strength: price vs moving averages
        ma20 = feat_df['WTI_Close'].rolling(20).mean()
        ma50 = feat_df['WTI_Close'].rolling(50).mean()
        sf['trend_20'] = (feat_df['WTI_Close'] - ma20) / (ma20 + 1e-8)
        sf['trend_50'] = (feat_df['WTI_Close'] - ma50) / (ma50 + 1e-8)

        # 3. RSI (14-day)
        delta = feat_df['WTI_Close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        sf['rsi_14'] = (rs / (1 + rs)) - 0.5  # center around 0, range [-0.5, 0.5]

        # 4. Volatility features
        log_ret = np.log(feat_df['WTI_Close'] / feat_df['WTI_Close'].shift(1))
        rv_10 = log_ret.rolling(10).std() * np.sqrt(252)
        rv_20 = log_ret.rolling(20).std() * np.sqrt(252)
        rv_60 = log_ret.rolling(60).std() * np.sqrt(252)
        sf['vol_ratio'] = rv_10 / (rv_60 + 1e-8) - 1  # vol expansion/contraction

        # 5. Spread z-score (WTI-Brent)
        spread = feat_df['WTI_Close'] - feat_df['Brent_Close']
        sp_mu  = spread.rolling(20).mean()
        sp_sig = spread.rolling(20).std()
        sf['spread_z'] = (spread - sp_mu) / (sp_sig + 1e-8)

        # 6. OVX change (implied vol sentiment)
        if 'OVX' in feat_df.columns:
            ovx_mu  = feat_df['OVX'].rolling(20).mean()
            ovx_sig = feat_df['OVX'].rolling(20).std()
            sf['ovx_z'] = (feat_df['OVX'] - ovx_mu) / (ovx_sig + 1e-8)
        else:
            sf['ovx_z'] = 0.0

        # 7. USD strength
        if 'USD_Index' in feat_df.columns:
            usd_mu  = feat_df['USD_Index'].rolling(20).mean()
            usd_sig = feat_df['USD_Index'].rolling(20).std()
            sf['usd_z'] = (feat_df['USD_Index'] - usd_mu) / (usd_sig + 1e-8)
        else:
            sf['usd_z'] = 0.0

        # 8. Net speculative position z-score
        if 'Net_Speculative_Position' in feat_df.columns:
            nsp_mu  = feat_df['Net_Speculative_Position'].rolling(20).mean()
            nsp_sig = feat_df['Net_Speculative_Position'].rolling(20).std()
            sf['nsp_z'] = (feat_df['Net_Speculative_Position'] - nsp_mu) / (nsp_sig + 1e-8)
        else:
            sf['nsp_z'] = 0.0

        # Clean
        sf = sf.replace([np.inf, -np.inf], np.nan)

        self.feature_names = list(sf.columns)
        return sf

    def _get_phi(self, state_features_row, position):
        """
        Build feature vector φ(s) for function approximation.
        Includes raw features + position encoding + interaction terms.
        """
        raw = state_features_row.values.astype(float)

        # Position encoding
        pos_enc = np.array([position, float(position == 1), float(position == -1)])

        # Interaction: momentum × position (captures if aligned)
        mom_pos = np.array([raw[1] * position])  # mom_10d × position

        phi = np.concatenate([raw, pos_enc, mom_pos, [1.0]])  # +1 bias
        return np.nan_to_num(phi, nan=0.0)

    def _q_value(self, phi, action):
        """Compute Q(s,a) = w_a · φ(s)."""
        return float(np.dot(self.weights[action], phi))

    def _softmax_policy(self, phi, tau):
        """Softmax/Boltzmann action selection."""
        q_vals = np.array([self._q_value(phi, a) for a in self.ACTIONS])

        # Numerical stability
        q_vals = q_vals - q_vals.max()
        exp_q = np.exp(q_vals / max(tau, 0.01))
        probs = exp_q / (exp_q.sum() + 1e-10)

        return probs

    def _differential_sharpe_reward(self, ret, action, position):
        """
        Differential Sharpe Ratio (Moody & Saffell, 1998).
        Provides a reward that directly optimizes the Sharpe ratio.
        """
        R_t = action * ret

        # Transaction cost
        if action != position:
            R_t -= self.transaction_cost

        # Update EMAs
        dA = R_t - self._A
        dB = R_t ** 2 - self._B

        # Differential Sharpe
        denom = (self._B - self._A ** 2)
        if denom > 1e-10:
            dS = (self._B * dA - 0.5 * self._A * dB) / (denom ** 1.5 + 1e-10)
        else:
            dS = R_t  # fallback to simple return when no variance yet

        # Update EMAs
        self._A += self.eta * dA
        self._B += self.eta * dB

        return dS

    # ── FEATURE ENGINEERING ──────────────────────────────────────────────
    def engineer_features(self, master_df):
        print("  [1/4] Engineering features for RL agent...")
        feat = master_df.copy()

        # Basic derived features (needed for state building)
        feat['WTI_LogRet_1d'] = np.log(feat['WTI_Close'] / feat['WTI_Close'].shift(1))
        feat['WTI_Mom_10']    = feat['WTI_Close'].pct_change(10)
        feat['WTI_Mom_20']    = feat['WTI_Close'].pct_change(20)
        feat['RealizedVol_20d'] = feat['WTI_LogRet_1d'].rolling(20).std() * np.sqrt(252)

        feat['Spread'] = feat['WTI_Close'] - feat['Brent_Close']
        spread_mu  = feat['Spread'].rolling(20).mean()
        spread_sig = feat['Spread'].rolling(20).std()
        feat['Spread_Zscore'] = (feat['Spread'] - spread_mu) / (spread_sig + 1e-8)

        if 'OVX' in feat.columns:
            feat['OVX_Chg_5d'] = feat['OVX'].pct_change(5)

        if 'USD_Index' in feat.columns:
            feat['USD_Chg_5d'] = feat['USD_Index'].pct_change(5)

        if 'Net_Speculative_Position' in feat.columns:
            nsp_mu  = feat['Net_Speculative_Position'].rolling(20).mean()
            nsp_sig = feat['Net_Speculative_Position'].rolling(20).std()
            feat['NSP_Zscore'] = (feat['Net_Speculative_Position'] - nsp_mu) / (nsp_sig + 1e-8)

        if 'Crack_3_2_1' in feat.columns:
            crack_mu  = feat['Crack_3_2_1'].rolling(20).mean()
            crack_sig = feat['Crack_3_2_1'].rolling(20).std()
            feat['Crack_Zscore'] = (feat['Crack_3_2_1'] - crack_mu) / (crack_sig + 1e-8)

        # Clean
        feat = feat.replace([np.inf, -np.inf], np.nan).dropna()

        # Drop zero-variance
        zero_var = [c for c in feat.columns if feat[c].std() == 0]
        if zero_var:
            feat = feat.drop(columns=zero_var)

        print(f"    ✓ Clean DataFrame: {feat.shape[0]} rows × {feat.shape[1]} cols")
        return feat

    # ── TD(0) SEMI-GRADIENT LEARNING ─────────────────────────────────────
    def fit_predict(self, feat_df):
        feat = feat_df.copy()

        print("  [2/4] Semi-gradient TD(0) with linear function approximation...")

        # Build state feature matrix
        state_features = self._build_state_features(feat)
        print(f"    State features: {len(self.feature_names)} continuous dimensions")
        print(f"    Features: {self.feature_names}")

        # Daily returns
        feat['Daily_Return'] = feat['WTI_Close'].pct_change().fillna(0)
        daily_ret = feat['Daily_Return'].values

        # Phi dimensionality: features + position(3) + interaction(1) + bias(1)
        sample_phi = self._get_phi(state_features.iloc[self.initial_train_days], 0)
        self.n_features = len(sample_phi)
        print(f"    φ(s) dimension: {self.n_features}")

        # Initialize weights with small random values
        np.random.seed(42)
        for a in self.ACTIONS:
            self.weights[a] = np.random.randn(self.n_features) * 0.001

        n = len(feat)
        signals = np.zeros(n)
        probabilities = np.full(n, np.nan)

        # ── Phase 1: Train on initial window (multiple epochs) ───────────
        print(f"    Phase 1: Training {self.n_epochs} epochs on "
              f"{self.initial_train_days} days...")

        total_updates = 0
        for epoch in range(self.n_epochs):
            position = 0
            self._A = 0.0
            self._B = 0.0
            equity = 1.0
            peak_equity = 1.0
            epoch_reward = 0.0

            # Decay learning rate across epochs
            lr = self.alpha * (1.0 / (1.0 + 0.3 * epoch))
            tau = self.tau_start * (0.7 ** epoch)  # cool temperature each epoch

            for t in range(60, self.initial_train_days):
                phi = self._get_phi(state_features.iloc[t], position)

                # Softmax action selection
                probs = self._softmax_policy(phi, tau)
                action = np.random.choice(self.ACTIONS, p=probs)

                # Differential Sharpe reward
                reward = self._differential_sharpe_reward(
                    daily_ret[t], action, position
                )

                # Drawdown penalty
                equity *= (1 + action * daily_ret[t])
                peak_equity = max(peak_equity, equity)
                dd = (equity - peak_equity) / (peak_equity + 1e-8)
                if dd < -0.05:  # penalize when drawdown > 5%
                    reward += self.drawdown_penalty * dd

                epoch_reward += reward

                # Next state
                if t + 1 < self.initial_train_days:
                    next_phi = self._get_phi(state_features.iloc[t + 1], action)
                    next_q_max = max(self._q_value(next_phi, a) for a in self.ACTIONS)
                else:
                    next_q_max = 0.0

                # Semi-gradient TD(0) update:
                # w_a ← w_a + α[r + γ·max_a' Q(s',a') - Q(s,a)] · φ(s)
                current_q = self._q_value(phi, action)
                td_error = reward + self.gamma * next_q_max - current_q
                self.weights[action] += lr * td_error * phi

                # L2 regularization to prevent weight explosion
                for a in self.ACTIONS:
                    self.weights[a] *= 0.9999

                position = action
                total_updates += 1

            avg_r = epoch_reward / (self.initial_train_days - 60)
            print(f"      Epoch {epoch+1}/{self.n_epochs} | "
                  f"lr={lr:.4f} τ={tau:.3f} | "
                  f"avg_reward={avg_r:.6f} | equity={equity:.3f}")

        print(f"    ✓ {total_updates} weight updates completed")

        # ── Phase 2: Walk-forward OOS with online learning ───────────────
        print(f"    Phase 2: Walk-forward from day {self.initial_train_days}...")

        position = 0
        self._A = 0.0
        self._B = 0.0
        equity = 1.0
        peak_equity = 1.0
        self.fold_metrics = []

        # Target for accuracy tracking
        fwd_ret = feat['WTI_Close'].shift(-5) / feat['WTI_Close'] - 1
        feat['Target'] = (fwd_ret > 0).astype(int)

        fold_rewards = []
        fold_start_t = self.initial_train_days

        for t in range(self.initial_train_days, n):
            phi = self._get_phi(state_features.iloc[t], position)

            # Decay temperature over OOS period
            progress = (t - self.initial_train_days) / max(n - self.initial_train_days, 1)
            tau = self.tau_end + (self.tau_start * 0.5 - self.tau_end) * (1 - progress)

            # Softmax action selection
            probs = self._softmax_policy(phi, tau)
            action = np.random.choice(self.ACTIONS, p=probs)
            signals[t] = action

            # Store probability of chosen action as confidence
            action_idx = self.ACTIONS.index(action)
            probabilities[t] = probs[0]  # P(buy)

            # Reward & online TD update
            reward = self._differential_sharpe_reward(
                daily_ret[t], action, position
            )

            # Drawdown penalty
            equity *= (1 + action * daily_ret[t])
            peak_equity = max(peak_equity, equity)
            dd = (equity - peak_equity) / (peak_equity + 1e-8)
            if dd < -0.05:
                reward += self.drawdown_penalty * dd

            fold_rewards.append(reward)

            # Online TD update (reduced learning rate)
            online_lr = self.alpha * 0.3
            if t + 1 < n:
                next_phi = self._get_phi(state_features.iloc[t + 1], action)
                next_q_max = max(self._q_value(next_phi, a) for a in self.ACTIONS)
            else:
                next_q_max = 0.0

            current_q = self._q_value(phi, action)
            td_error = reward + self.gamma * next_q_max - current_q
            self.weights[action] += online_lr * td_error * phi

            for a in self.ACTIONS:
                self.weights[a] *= 0.9999

            position = action

            # Record fold metrics periodically
            if (t - fold_start_t + 1) % self.retrain_every == 0 and fold_rewards:
                fold_avg = np.mean(fold_rewards)
                self.fold_metrics.append({
                    'fold': len(self.fold_metrics) + 1,
                    'train': t, 'test': self.retrain_every,
                    'period': f"{feat.index[t - self.retrain_every + 1].date()} → "
                              f"{feat.index[t].date()}",
                    'acc': float(np.mean([r > 0 for r in fold_rewards[-self.retrain_every:]]))
                })
                fold_rewards = []

        feat['Signal'] = signals.astype(int)
        feat['Signal_Label'] = feat['Signal'].map({1: 'BUY', -1: 'SELL', 0: 'HOLD'})
        feat['Probability'] = probabilities
        feat['Prediction'] = np.where(probabilities > 0.5, 1.0, 0.0)

        # NaN before OOS
        feat.loc[feat.index[:self.initial_train_days], 'Prediction'] = np.nan
        feat.loc[feat.index[:self.initial_train_days], 'Probability'] = np.nan

        # Regime labeling (by volatility tercile)
        rv = feat['RealizedVol_20d']
        q33, q66 = rv.quantile(0.33), rv.quantile(0.66)
        feat['Regime'] = 'CHOPPY'
        feat.loc[rv <= q33, 'Regime'] = 'BULL'
        feat.loc[rv >= q66, 'Regime'] = 'PANIC'

        # ── Feature importance from weight magnitudes ────────────────────
        avg_weights = sum(np.abs(self.weights[a]) for a in self.ACTIONS) / len(self.ACTIONS)

        # Map weights back to feature names
        raw_feat_names = self.feature_names.copy()
        all_names = raw_feat_names + ['position', 'is_long', 'is_short',
                                       'mom_x_pos', 'bias']
        importance_dict = {}
        for i, name in enumerate(all_names):
            if i < len(avg_weights):
                importance_dict[name] = float(avg_weights[i])

        self.feature_importance = pd.Series(importance_dict).sort_values(ascending=False)
        self.feature_cols = list(self.feature_importance.index)

        # ── Summary ──────────────────────────────────────────────────────
        oos = feat.iloc[self.initial_train_days:]
        n_buy  = (oos['Signal'] == 1).sum()
        n_sell = (oos['Signal'] == -1).sum()
        n_hold = (oos['Signal'] == 0).sum()
        total  = len(oos)

        fold_df = pd.DataFrame(self.fold_metrics) if self.fold_metrics else pd.DataFrame()
        avg_acc = fold_df['acc'].mean() if len(fold_df) > 0 else 0

        print(f"\n    ✓ φ(s) dim: {self.n_features} | "
              f"Weight norms: " + " | ".join(
                  f"a={a}: {np.linalg.norm(self.weights[a]):.4f}" for a in self.ACTIONS))
        print(f"    ✓ {len(fold_df)} folds | Avg positive reward rate: {avg_acc:.1%}")
        print(f"    ✓ Signals: BUY={n_buy} ({n_buy/total*100:.1f}%) | "
              f"SELL={n_sell} ({n_sell/total*100:.1f}%) | "
              f"HOLD={n_hold} ({n_hold/total*100:.1f}%)")

        self.metadata.update({
            'avg_accuracy': avg_acc,
            'n_features_phi': self.n_features,
            'weight_norms': {a: float(np.linalg.norm(self.weights[a]))
                             for a in self.ACTIONS},
            'has_regimes': True,
            'fwd_days': 5,
        })

        return feat

    # ── MULTI-HORIZON FORECASTING ────────────────────────────────────────
    def forecast_returns(self, feat_df):
        """
        Multi-horizon return forecast using the learned RL value function.
        For each horizon, estimates expected return direction from the current
        state's Q-values combined with historical conditional returns.
        """
        print(f"  [4/4] Multi-horizon return forecasting ({self.forecast_horizons})...")
        feat = feat_df.copy()

        # Build state features for the last observation
        state_features = self._build_state_features(feat)
        last_phi = self._get_phi(state_features.iloc[-1], 0)  # neutral position

        # Q-value based directional confidence
        q_buy  = self._q_value(last_phi, 1)
        q_sell = self._q_value(last_phi, -1)
        q_hold = self._q_value(last_phi, 0)
        q_total = abs(q_buy) + abs(q_sell) + abs(q_hold) + 1e-8
        rl_bullish = (q_buy - q_sell) / q_total  # [-1, 1] bias

        forecasts = {}
        for horizon in self.forecast_horizons:
            fwd = feat['WTI_Close'].shift(-horizon) / feat['WTI_Close'] - 1
            valid = fwd.dropna()

            if len(valid) < 200:
                continue

            # Combine RL signal with historical conditional analysis
            recent_mom = feat['WTI_Mom_10'].iloc[-1]
            recent_vol = feat['RealizedVol_20d'].iloc[-1]

            # Historical: similar momentum regime
            similar = feat['WTI_Mom_10'].between(recent_mom - 0.03, recent_mom + 0.03)
            hist_fwd = fwd[similar].dropna()

            if len(hist_fwd) > 20:
                hist_prob_up = float((hist_fwd > 0).mean())
                hist_exp_ret = float(hist_fwd.mean())
            else:
                hist_prob_up = 0.5
                hist_exp_ret = float(valid.mean())

            # Blend RL confidence with historical
            rl_prob_up = 0.5 + 0.3 * np.tanh(rl_bullish)  # map to [0.2, 0.8]
            prob_up = 0.4 * rl_prob_up + 0.6 * hist_prob_up

            # Scale expected return by horizon
            exp_ret = hist_exp_ret * (1 + 0.2 * np.sign(rl_bullish))

            forecasts[horizon] = {
                'prob_up': float(np.clip(prob_up, 0.05, 0.95)),
                'expected_return': float(exp_ret),
            }
            direction = "▲" if prob_up > 0.5 else "▼"
            print(f"    ✓ {horizon:3d}d → P(up)={prob_up:.1%} | "
                  f"E[ret]={exp_ret:+.2%} {direction}")

        self.metadata['forecasts'] = forecasts
        return feat
