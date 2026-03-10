# ============================================================================
# PETROQUANT — MODULAR STRATEGY FRAMEWORK
# ============================================================================
# Contains:
#   BaseStrategy       — abstract base class for all strategies
#   VolatilityEngine   — GARCH + ATR + OVX composite volatility CI
#   HMMXGBoostStrategy — HMM regime detection + XGBoost walk-forward
#
# USAGE:
#   from strategy import HMMXGBoostStrategy
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
from hmmlearn.hmm import GaussianHMM
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')


# ═════════════════════════════════════════════════════════════════════════════
# VOLATILITY ENGINE — 3-LAYER COMPOSITE CI
# ═════════════════════════════════════════════════════════════════════════════
class VolatilityEngine:
    """
    Computes horizon-aware 95% CI using 3 volatility layers:
      1. GARCH(1,1) — conditional volatility with mean-reversion
      2. ATR(14)     — gap-aware, price-action-sensitive noise floor
      3. OVX (IV)    — market's own implied volatility (CBOE Oil VIX)

    Blended with horizon-dependent weights:
      Short (1-7d):   heavier ATR   (price-action sensitive)
      Medium (15-30d): heavier GARCH (clustering + mean-reversion)
      Long (60-180d):  heavier IV    (market's forward view)
    """

    # Horizon-dependent blend weights: {horizon: (w_garch, w_atr, w_iv)}
    BLEND_WEIGHTS = {
        1:   (0.25, 0.50, 0.25),
        7:   (0.30, 0.40, 0.30),
        15:  (0.40, 0.25, 0.35),
        30:  (0.40, 0.20, 0.40),
        60:  (0.30, 0.15, 0.55),
        90:  (0.25, 0.10, 0.65),
        180: (0.20, 0.10, 0.70),
    }

    def __init__(self, prices, ovx_series=None, lookback=252):
        """
        Parameters
        ----------
        prices    : pd.Series of daily close prices (WTI_Close)
        ovx_series: pd.Series of OVX values (optional, from pipeline)
        lookback  : int, number of days for GARCH fitting
        """
        self.prices = prices.dropna()
        self.ovx = ovx_series
        self.lookback = lookback

        # Pre-compute layers
        self._fit_garch()
        self._compute_atr()
        self._compute_iv()

    # ── Layer 1: GARCH(1,1) ──────────────────────────────────────────
    def _fit_garch(self):
        """Fit GARCH(1,1) on recent daily log-returns."""
        from arch import arch_model

        log_ret = np.log(self.prices / self.prices.shift(1)).dropna()
        recent = log_ret.tail(self.lookback) * 100  # arch expects percent returns

        try:
            model = arch_model(recent, vol='Garch', p=1, q=1, mean='Zero',
                               rescale=False)
            res = model.fit(disp='off', show_warning=False)

            # Extract GARCH parameters
            self.omega = res.params.get('omega', 0.01)
            self.alpha = res.params.get('alpha[1]', 0.05)
            self.beta  = res.params.get('beta[1]', 0.90)
            # Last conditional variance (in %^2)
            self.sigma2_1 = float(res.conditional_volatility.iloc[-1] ** 2)
            self.garch_fitted = True
            # Long-run variance
            persist = self.alpha + self.beta
            if persist < 1.0:
                self.sigma2_lr = self.omega / (1.0 - persist)
            else:
                self.sigma2_lr = self.sigma2_1
        except Exception:
            # Fallback to simple realized vol
            daily_vol = log_ret.tail(60).std()
            self.sigma2_1 = (daily_vol * 100) ** 2
            self.sigma2_lr = self.sigma2_1
            self.alpha = 0.05
            self.beta = 0.90
            self.omega = self.sigma2_lr * 0.05
            self.garch_fitted = False

    def _garch_horizon_vol(self, horizon):
        """
        GARCH h-step ahead annualized volatility (in decimal, not %).
        σ²(h) = h·σ²_lr + (α+β)·[(1-(α+β)^h)/(1-(α+β))]·(σ²₁ - σ²_lr)
        """
        persist = self.alpha + self.beta
        if abs(persist - 1.0) < 1e-6:
            # Unit-root IGARCH: variance scales linearly
            total_var = self.sigma2_1 * horizon
        else:
            # Mean-reverting GARCH variance forecast
            total_var = (horizon * self.sigma2_lr +
                         persist * (1 - persist**horizon) / (1 - persist) *
                         (self.sigma2_1 - self.sigma2_lr))
        # Convert from %^2 to decimal
        return np.sqrt(max(total_var, 1e-8)) / 100.0

    # ── Layer 2: ATR(14) ─────────────────────────────────────────────
    def _compute_atr(self):
        """Compute 14-day ATR using daily close-to-close as proxy for True Range."""
        close = self.prices
        # True Range proxy: use daily absolute return * close (no High/Low available)
        daily_abs_change = (close - close.shift(1)).abs()
        # EMA smoothing (14-day)
        self.atr_14 = float(daily_abs_change.ewm(span=14, adjust=False).mean().iloc[-1])
        self.current_price = float(close.iloc[-1])

    def _atr_horizon_vol(self, horizon):
        """
        ATR-based volatility scaled to horizon.
        Uses sqrt(h) scaling with a decay factor to prevent long-horizon explosion.
        Returns annualized-style volatility as decimal fraction of price.
        """
        # ATR as fraction of price
        atr_frac = self.atr_14 / self.current_price
        # Scale: sqrt(h) with decay factor 0.9^(h/5)
        decay = 0.9 ** (horizon / 5.0)
        return atr_frac * np.sqrt(horizon) * decay

    # ── Layer 3: OVX Implied Volatility ──────────────────────────────
    def _compute_iv(self):
        """Get the latest OVX value (annualized implied vol for crude oil)."""
        if self.ovx is not None and len(self.ovx.dropna()) > 0:
            self.ovx_latest = float(self.ovx.dropna().iloc[-1])
        else:
            self.ovx_latest = None

    def _iv_horizon_vol(self, horizon):
        """
        OVX-based volatility scaled to horizon.
        OVX is annualized → σ_iv(h) = (OVX/100) * sqrt(h/252)
        """
        if self.ovx_latest is None:
            return None
        return (self.ovx_latest / 100.0) * np.sqrt(horizon / 252.0)

    # ── Composite Blending ───────────────────────────────────────────
    def compute_ci(self, horizon, expected_return):
        """
        Compute 95% CI for a given horizon.

        Returns
        -------
        dict with: ci_lower, ci_upper, price_target, vol_garch, vol_atr,
                   vol_iv, vol_composite, current_price
        """
        vol_garch = self._garch_horizon_vol(horizon)
        vol_atr   = self._atr_horizon_vol(horizon)
        vol_iv    = self._iv_horizon_vol(horizon)

        # Get blend weights (fall back to equal if horizon not in map)
        w_g, w_a, w_i = self.BLEND_WEIGHTS.get(horizon, (0.33, 0.33, 0.34))

        # If IV unavailable, redistribute its weight to GARCH and ATR
        if vol_iv is None:
            total = w_g + w_a
            w_g, w_a, w_i = w_g / total, w_a / total, 0.0
            vol_iv_val = 0.0
        else:
            vol_iv_val = vol_iv

        vol_composite = w_g * vol_garch + w_a * vol_atr + w_i * vol_iv_val

        price_target = self.current_price * (1 + expected_return)
        ci_lower = self.current_price * (1 + expected_return - 1.96 * vol_composite)
        ci_upper = self.current_price * (1 + expected_return + 1.96 * vol_composite)

        return {
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'price_target': float(price_target),
            'vol_garch': float(vol_garch),
            'vol_atr': float(vol_atr),
            'vol_iv': float(vol_iv_val) if vol_iv is not None else None,
            'vol_composite': float(vol_composite),
            'current_price': float(self.current_price),
        }

    def print_summary(self):
        """Print a summary of the volatility engine state."""
        garch_status = "[OK] GARCH(1,1) fitted" if self.garch_fitted else "[WARN] GARCH fallback (realized vol)"
        print(f"    {garch_status}")
        print(f"      alpha={self.alpha:.4f}  beta={self.beta:.4f}  "
              f"persistence={self.alpha+self.beta:.4f}")
        print(f"      1-day cond. vol = {np.sqrt(self.sigma2_1)/100:.4f} "
              f"({np.sqrt(self.sigma2_1):.2f}%)")

        print(f"    [OK] ATR(14) = ${self.atr_14:.2f} "
              f"({self.atr_14/self.current_price*100:.2f}% of price)")

        if self.ovx_latest is not None:
            print(f"    [OK] OVX (Implied Vol) = {self.ovx_latest:.1f}% annualized")
        else:
            print(f"    [WARN] OVX not available -- using GARCH + ATR only")


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
        """Full pipeline: features -> fit/predict -> forecast -> return enriched df."""
        print(f"\n{'='*70}")
        print(f"  RUNNING STRATEGY: {self.name}")
        print(f"{'='*70}\n")

        feat_df = self.engineer_features(master_df)
        feat_df = self.fit_predict(feat_df)
        feat_df = self.forecast_returns(feat_df)

        print(f"\n{'='*70}")
        print(f"  [OK] {self.name} COMPLETE -- {len(feat_df)} rows with signals")
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
        print("         (Transforming raw market data into patterns the AI can learn from)")
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
        print(f"    [OK] {len(engineered)} engineered features from {len(raw_cols)} raw variables")
        print(f"      (e.g., 'WTI_LogRet_1d' = how much oil price changed today in %)")
        print(f"    [OK] Clean DataFrame: {feat.shape[0]} rows x {feat.shape[1]} cols")
        print(f"      (Each row = one trading day, each column = one data signal)")

        return feat

    # ── STEP 2+3: HMM + XGBOOST FIT/PREDICT ─────────────────────────────
    def fit_predict(self, feat_df):
        feat = feat_df.copy()

        # ── HMM Regime Detection ─────────────────────────────────────────
        print("  [2/4] Fitting HMM regime detection...")
        print("         (Detecting the market's 'mood' -- is it calm, choppy, or panicking?)")
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

        print(f"    [OK] HMM converged -- Log-likelihood: {hmm_model.score(X_hmm_scaled):.2f}")
        print(f"      (The model successfully learned 3 market moods from price patterns)")
        print()
        print(f"      {'Regime':<10} {'Avg Daily Return':>18} {'Avg Volatility':>16} {'Days':>12}")
        print(f"      {'─'*60}")
        for _, row in state_stats.iterrows():
            s = int(row['State'])
            name = regime_map[s]
            pct = row['Count'] / len(feat) * 100
            marker = {'BULL': '[+]', 'CHOPPY': '[~]', 'PANIC': '[!]'}.get(name, '[ ]')
            print(f"      {marker} {name:7s}   {row['Mean_Return']:+.5f} ({row['Mean_Return']*25200:+.1f}%/yr)"
                  f"    {row['Mean_Vol']:.4f}         "
                  f"{int(row['Count'])} days ({pct:.1f}%)")
        print()
        print(f"      * BULL = calm market trending up | CHOPPY = uncertain sideways")
        print(f"        PANIC = high fear, big swings (think COVID crash or oil war)")

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
        print(f"         (Training an AI on past data, then testing on unseen future data)")
        print(f"         (Retrains every {self.retrain_every} days to adapt to changing markets)")

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
                'period': f"{dates[i].date()} -> {dates[end-1].date()}",
                'acc': acc
            })
            i = end

        feat['Probability'] = probabilities
        feat['Prediction']  = predictions

        # Feature importance from last model
        self.feature_importance = pd.Series(
            xgb_model.feature_importances_, index=self.feature_cols
        ).sort_values(ascending=False)

        # ── Signals + Regime-Aware Position Sizing ────────────────────────
        mask = feat['Probability'].notna()
        feat['Signal'] = 0
        feat.loc[mask & (feat['Probability'] > self.buy_threshold), 'Signal'] = 1
        feat.loc[mask & (feat['Probability'] < self.sell_threshold), 'Signal'] = -1
        feat['Signal_Label'] = feat['Signal'].map({1: 'BUY', -1: 'SELL', 0: 'HOLD'})

        # Position sizing: scale by regime stability AND model confidence
        #   BULL  = calm, high conviction trades get full size
        #   CHOPPY = uncertain, reduce size to 50%
        #   PANIC  = volatile, reduce size to 25% (protect capital)
        regime_mult = {'BULL': 1.0, 'CHOPPY': 0.5, 'PANIC': 0.25}
        feat['Regime_Mult'] = feat['Regime'].map(regime_mult).fillna(0.5)

        # Confidence factor: higher probability distance from 0.5 = more conviction
        feat['Confidence_Factor'] = np.where(
            feat['Probability'].notna(),
            0.5 + 0.5 * (np.abs(feat['Probability'] - 0.5) * 2),  # scales 0.5-1.0
            0.0
        )

        # Final position size = signal direction * regime * confidence
        feat['Position_Size'] = feat['Signal'] * feat['Regime_Mult'] * feat['Confidence_Factor']

        fold_df = pd.DataFrame(self.fold_metrics)
        avg_acc = fold_df['acc'].mean()
        print(f"    [OK] {len(fold_df)} folds | Avg OOS accuracy: {avg_acc:.1%}")
        print(f"      (Out-of-sample = tested on data the model never saw during training)")
        acc_grade = 'EXCELLENT' if avg_acc > 0.58 else 'GOOD' if avg_acc > 0.54 else 'FAIR' if avg_acc > 0.50 else 'WEAK'
        print(f"      ({acc_grade}: >50% means the model is better than a coin flip)")

        oos = feat[mask]
        n_buy  = (oos['Signal'] == 1).sum()
        n_sell = (oos['Signal'] == -1).sum()
        n_hold = (oos['Signal'] == 0).sum()
        total  = len(oos)
        print(f"    [OK] Signals: BUY={n_buy} ({n_buy/total*100:.1f}%) | "
              f"SELL={n_sell} ({n_sell/total*100:.1f}%) | "
              f"HOLD={n_hold} ({n_hold/total*100:.1f}%)")
        print(f"      (BUY = model thinks price goes up | SELL = down | HOLD = not sure enough)")
        print(f"      (Threshold: BUY when Confidence > {self.buy_threshold:.0%}, "
              f"SELL when < {self.sell_threshold:.0%})")

        # Position sizing summary
        avg_pos = oos['Position_Size'].abs().mean()
        print(f"    [OK] Regime-Aware Position Sizing active")
        print(f"      Avg position size: {avg_pos:.2f} (1.0 = full, 0.25 = quarter)")
        print(f"      (Sized down in PANIC/CHOPPY regimes, sized up in BULL with high confidence)")

        self.metadata.update({
            'avg_accuracy': avg_acc,
            'n_folds': len(fold_df),
            'n_features': len(self.feature_cols),
            'fwd_days': self.fwd_days,
            'has_regimes': True,
            'position_sizing': 'regime_aware',
        })

        return feat

    # ── STEP 4: MULTI-HORIZON FORECASTING WITH VOLATILITY ENGINE ─────────
    def forecast_returns(self, feat_df):
        print(f"  [4/4] Multi-horizon return forecasting ({self.forecast_horizons})...")
        print(f"         (Predicting where WTI price will be in 1 day to 6 months)")
        print(f"         (Uses GARCH + ATR + OVX composite volatility for 95% CI)")
        feat = feat_df.copy()

        if not self.feature_cols:
            print("    [WARN] No feature columns -- skipping forecasts")
            return feat

        X_all = feat[self.feature_cols].values
        scaler = StandardScaler()
        forecasts = {}

        # ── Initialize Volatility Engine ─────────────────────────────────
        ovx_series = feat['OVX'] if 'OVX' in feat.columns else None
        vol_engine = VolatilityEngine(
            prices=feat['WTI_Close'],
            ovx_series=ovx_series,
            lookback=252
        )

        current_price = float(feat['WTI_Close'].iloc[-1])

        print()
        print(f"    Current WTI Price: ${current_price:.2f}")
        print()
        print(f"    -- Volatility Engine Layers --")
        vol_engine.print_summary()
        print()
        print(f"    {'Horizon':<10} {'Price Range (95% CI)':<28} {'Expected':<14} "
              f"{'Return':<12} {'Vol(G/A/I->C)':<28} {'Confidence':<14} {'View'}")
        print(f"    {'─'*120}")

        for horizon in self.forecast_horizons:
            # Build target: forward return for this horizon
            fwd = feat['WTI_Close'].shift(-horizon) / feat['WTI_Close'] - 1
            valid_mask = fwd.notna()

            if valid_mask.sum() < self.initial_train_days + 50:
                print(f"    [WARN] {horizon}d -- not enough data, skipping")
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

            # ── Compute volatility-aware CI ──────────────────────────────
            ci = vol_engine.compute_ci(horizon, exp_return)

            forecasts[horizon] = {
                'prob_up': float(prob_up),
                'expected_return': exp_return,
                'expected_price': ci['price_target'],
                'current_price': ci['current_price'],
                'price_low': ci['ci_lower'],
                'price_high': ci['ci_upper'],
                'vol_garch': ci['vol_garch'],
                'vol_atr': ci['vol_atr'],
                'vol_iv': ci['vol_iv'],
                'vol_composite': ci['vol_composite'],
            }

            direction = "^ Bullish" if prob_up > 0.55 else "v Bearish" if prob_up < 0.45 else "- Neutral"
            conf_label = "HIGH" if abs(prob_up - 0.5) > 0.15 else "MEDIUM" if abs(prob_up - 0.5) > 0.05 else "LOW"

            # Format vol breakdown
            vg = ci['vol_garch'] * 100
            va = ci['vol_atr'] * 100
            vi = ci['vol_iv'] * 100 if ci['vol_iv'] is not None else 0
            vc = ci['vol_composite'] * 100
            vol_str = f"{vg:.1f}/{va:.1f}/{vi:.1f}->{vc:.1f}%"

            range_str = f"${ci['ci_lower']:.2f} -- ${ci['ci_upper']:.2f}"
            pad = max(28 - len(range_str), 0)
            print(f"    {horizon:3d}d       {range_str}{' '*pad}"
                  f"${ci['price_target']:<10.2f}    {exp_return:>+8.2%}     "
                  f"{vol_str:<28s}"
                  f"{prob_up:.1%} ({conf_label})"
                  f"{'':>4}{direction}")

        print(f"    {'─'*120}")
        print(f"    * 95% CI = composite of GARCH(1,1) + ATR(14) + OVX implied vol")
        print(f"      G=GARCH conditional vol | A=ATR noise floor | I=OVX implied vol | C=Composite blend")
        print(f"      Short horizons weight ATR heavily, long horizons weight IV heavily")
        print(f"    * Confidence = model's estimated probability that price goes UP")
        print(f"      HIGH (>65% or <35%) = strong conviction | LOW (45-55%) = uncertain")

        self.metadata['forecasts'] = forecasts
        return feat
