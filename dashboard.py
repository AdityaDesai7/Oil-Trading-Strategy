# ============================================================================
# PETROQUANT — BACKTESTING ENGINE + STRATEGY DASHBOARD
# ============================================================================
# Contains:
#   Backtester        — computes all backtest metrics from strategy signals
#   StrategyDashboard — renders a premium 10-panel Plotly dashboard
#
# USAGE:
#   from dashboard import Backtester, StrategyDashboard
#   bt = Backtester()
#   result = bt.run(strategy_df)
#   dash = StrategyDashboard()
#   dash.render(result, strategy)
# ============================================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass, field
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')


# ═════════════════════════════════════════════════════════════════════════════
# BACKTEST RESULT CONTAINER
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class BacktestResult:
    """Container for all backtesting outputs."""
    oos_df: pd.DataFrame              # Out-of-sample DataFrame with returns
    total_return: float = 0.0
    bnh_return: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    calmar_ratio: float = 0.0
    total_trades: int = 0
    trading_days: int = 0
    annual_return: float = 0.0
    annual_vol: float = 0.0
    # Advanced ratios
    sortino: float = 0.0
    information_ratio: float = 0.0
    treynor_ratio: float = 0.0
    sterling_ratio: float = 0.0
    burke_ratio: float = 0.0
    expectancy: float = 0.0
    omega_ratio: float = 0.0
    tail_ratio: float = 0.0
    up_capture: float = 0.0
    down_capture: float = 0.0
    capture_ratio: float = 0.0
    ulcer_index: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0


# ═════════════════════════════════════════════════════════════════════════════
# BACKTESTER
# ═════════════════════════════════════════════════════════════════════════════
class Backtester:
    """Compute backtest metrics from strategy signals."""

    def run(self, strategy_df, price_col='WTI_Close'):
        """
        Run backtest on strategy output DataFrame.

        Parameters
        ----------
        strategy_df : pd.DataFrame with 'Signal', price_col columns
        price_col   : str, column name for the asset price

        Returns
        -------
        BacktestResult with all metrics and enriched OOS DataFrame
        """
        print("\n  -- Backtesting ------------------------------------------------------")
        print("     (Simulating: 'If I had followed these signals in the past, how would I have done?')")

        # Filter to out-of-sample period (where we have signals)
        has_signal = strategy_df['Probability'].notna() if 'Probability' in strategy_df.columns \
            else strategy_df['Signal'] != 0
        oos = strategy_df[has_signal].copy()

        if len(oos) == 0:
            print("    [WARN] No OOS data to backtest")
            return BacktestResult(oos_df=oos)

        # ── Returns ──────────────────────────────────────────────────
        oos['Daily_Return']      = oos[price_col].pct_change()
        oos['Strategy_Return']   = oos['Signal'].shift(1) * oos['Daily_Return']
        oos['Strategy_Return']   = oos['Strategy_Return'].fillna(0)
        oos['BnH_Return']        = oos['Daily_Return'].fillna(0)
        oos['Strategy_Cumulative'] = (1 + oos['Strategy_Return']).cumprod()
        oos['BnH_Cumulative']      = (1 + oos['BnH_Return']).cumprod()

        # ── Drawdown ─────────────────────────────────────────────────
        cum_max = oos['Strategy_Cumulative'].cummax()
        oos['Drawdown'] = (oos['Strategy_Cumulative'] - cum_max) / cum_max

        # ── Rolling Sharpe ───────────────────────────────────────────
        oos['Rolling_Sharpe'] = (
            oos['Strategy_Return'].rolling(60).mean() /
            (oos['Strategy_Return'].rolling(60).std() + 1e-8)
        ) * np.sqrt(252)

        # ── Monthly Returns ─────────────────────────────────────────
        oos['YearMonth'] = oos.index.to_period('M')

        # ═════════════════════════════════════════════════════════════
        # CORE METRICS
        # ═════════════════════════════════════════════════════════════
        strat_ret   = oos['Strategy_Return']
        bnh_daily   = oos['BnH_Return']
        total_ret   = oos['Strategy_Cumulative'].iloc[-1] - 1
        bnh_ret     = oos['BnH_Cumulative'].iloc[-1] - 1
        sharpe      = (strat_ret.mean() / (strat_ret.std() + 1e-8)) * np.sqrt(252)
        max_dd      = oos['Drawdown'].min()
        trades      = strat_ret[strat_ret != 0]
        win_rate    = (trades > 0).mean() if len(trades) > 0 else 0
        gross_p     = trades[trades > 0].sum()
        gross_l     = abs(trades[trades < 0].sum())
        pf          = gross_p / gross_l if gross_l > 0 else np.inf
        n_years     = len(oos) / 252
        annual_ret  = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1
        annual_vol  = strat_ret.std() * np.sqrt(252)
        calmar      = annual_ret / abs(max_dd) if max_dd != 0 else np.inf

        # ═════════════════════════════════════════════════════════════
        # ADVANCED RATIOS
        # ═════════════════════════════════════════════════════════════

        # 1. Sortino Ratio -- only penalizes downside volatility
        downside = strat_ret[strat_ret < 0]
        downside_std = downside.std() if len(downside) > 0 else 1e-8
        sortino = (strat_ret.mean() / (downside_std + 1e-8)) * np.sqrt(252)

        # 2. Information Ratio -- skill vs benchmark
        excess = strat_ret - bnh_daily
        tracking_error = excess.std() + 1e-8
        information_ratio = (excess.mean() / tracking_error) * np.sqrt(252)

        # 3. Treynor Ratio -- return per unit of systematic risk
        cov_matrix = np.cov(strat_ret.dropna().values, bnh_daily.dropna().values)
        beta = cov_matrix[0, 1] / (cov_matrix[1, 1] + 1e-8)
        treynor = annual_ret / (beta + 1e-8) if beta != 0 else np.inf

        # 4. Sterling Ratio -- return / average of top 3 drawdowns
        dd_series = oos['Drawdown']
        dd_periods = []
        in_dd = False
        current_dd = 0
        for dd_val in dd_series:
            if dd_val < 0:
                in_dd = True
                current_dd = min(current_dd, dd_val)
            elif in_dd:
                dd_periods.append(current_dd)
                current_dd = 0
                in_dd = False
        if in_dd:
            dd_periods.append(current_dd)
        top3_dd = sorted(dd_periods)[:3] if len(dd_periods) >= 3 else dd_periods
        avg_top3 = abs(np.mean(top3_dd)) if top3_dd else 1e-8
        sterling = annual_ret / avg_top3 if avg_top3 > 0 else np.inf

        # 5. Burke Ratio -- return / sqrt(sum of squared drawdowns)
        squared_dd_sum = sum(d ** 2 for d in dd_periods) if dd_periods else 1e-8
        burke = annual_ret / (np.sqrt(squared_dd_sum) + 1e-8)

        # 6. Expectancy -- expected $ per trade
        avg_win = trades[trades > 0].mean() if (trades > 0).any() else 0
        avg_loss = abs(trades[trades < 0].mean()) if (trades < 0).any() else 0
        loss_rate = 1 - win_rate
        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)

        # 7. Omega Ratio -- probability-weighted gains vs losses at threshold=0
        gains = strat_ret[strat_ret > 0].sum()
        losses = abs(strat_ret[strat_ret < 0].sum())
        omega = (gains / losses) if losses > 0 else np.inf

        # 8. Tail Ratio -- are big wins bigger than big losses?
        p95 = strat_ret.quantile(0.95)
        p05 = abs(strat_ret.quantile(0.05))
        tail = p95 / p05 if p05 > 0 else np.inf

        # 9. Up/Down Capture Ratios
        up_days = bnh_daily > 0
        dn_days = bnh_daily < 0
        up_capture = (strat_ret[up_days].mean() / bnh_daily[up_days].mean()) if up_days.sum() > 0 else 0
        down_capture = (strat_ret[dn_days].mean() / bnh_daily[dn_days].mean()) if dn_days.sum() > 0 else 0
        capture_ratio = up_capture / down_capture if down_capture != 0 else np.inf

        # 10. Ulcer Index -- depth & duration of drawdowns
        ulcer = np.sqrt((dd_series ** 2).mean()) * 100

        result = BacktestResult(
            oos_df=oos,
            total_return=total_ret,
            bnh_return=bnh_ret,
            sharpe=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=pf,
            calmar_ratio=calmar,
            total_trades=len(trades),
            trading_days=len(strat_ret),
            annual_return=annual_ret,
            annual_vol=annual_vol,
            sortino=sortino,
            information_ratio=information_ratio,
            treynor_ratio=treynor,
            sterling_ratio=sterling,
            burke_ratio=burke,
            expectancy=expectancy,
            omega_ratio=omega,
            tail_ratio=tail,
            up_capture=up_capture,
            down_capture=down_capture,
            capture_ratio=capture_ratio,
            ulcer_index=ulcer,
            avg_win=avg_win,
            avg_loss=avg_loss,
        )

        # ═════════════════════════════════════════════════════════════
        # PRETTY PRINT — CATEGORIZED PERFORMANCE REPORT
        # ═════════════════════════════════════════════════════════════
        def _grade(val, thresholds, reverse=False):
            """Return a letter grade based on thresholds: (excellent, good, fair)."""
            a, b, c = thresholds
            if reverse:
                if val <= a: return '[A+]'
                if val <= b: return '[B ]'
                if val <= c: return '[C ]'
                return '[D-]'
            else:
                if val >= a: return '[A+]'
                if val >= b: return '[B ]'
                if val >= c: return '[C ]'
                return '[D-]'

        W = 88  # total box width
        HR = '-' * W  # horizontal rule
        print()
        print(f"  +{HR}+")
        print(f"  |{'PETROQUANT -- STRATEGY PERFORMANCE REPORT':^{W}}|")
        print(f"  |{HR}|")
        print(f"  |{'(How well did the strategy perform if we had traded it in the past?)':^{W}}|")
        print(f"  +{HR}+")

        # ---- THE BASICS ----
        print(f"  |{'':^{W}}|")
        print(f"  |  >> THE BASICS (Return & Activity){'':>{W-37}}|")
        print(f"  |{HR}|")
        _p = lambda label, val, desc, grade='': print(
            f"  |  {label:<28} {val:>12}   {grade:<6} {desc:<{W-52}}|"
        )
        _p('Total Strategy Return', f'{total_ret*100:+.2f}%',
           'Total profit if you followed every signal', _grade(total_ret*100, (50, 20, 0)))
        _p('Buy & Hold Return', f'{bnh_ret*100:+.2f}%',
           'What you\'d earn just holding oil (the benchmark)', '')
        _p('Alpha (Strategy - B&H)', f'{(total_ret-bnh_ret)*100:+.2f}%',
           'Extra return the strategy generated over passive holding',
           _grade((total_ret-bnh_ret)*100, (30, 10, 0)))
        _p('Annualized Return', f'{annual_ret*100:+.2f}%',
           'Average yearly return, compounded', _grade(annual_ret*100, (20, 10, 0)))
        _p('Annualized Volatility', f'{annual_vol*100:.2f}%',
           'How much the returns swing per year (lower = smoother)', _grade(annual_vol*100, (15, 25, 35), reverse=True))
        _p('Trading Days', f'{len(strat_ret):,}',
           'Total calendar days in the backtest', '')
        _p('Total Trades', f'{len(trades):,}',
           'Days where we were actively long or short', '')

        # ---- RISK-ADJUSTED ----
        print(f"  |{'':^{W}}|")
        print(f"  |  >> RISK-ADJUSTED RATIOS (Return normalised by risk taken){'':>{W-59}}|")
        print(f"  |{HR}|")
        _p('Sharpe Ratio', f'{sharpe:.3f}',
           'Return / total volatility (>= 1.0 is good)', _grade(sharpe, (1.5, 1.0, 0.5)))
        _p('Sortino Ratio', f'{sortino:.3f}',
           'Like Sharpe, but only counts DOWNSIDE swings (>= 1.5 ideal)', _grade(sortino, (2.0, 1.0, 0.5)))
        _p('Information Ratio', f'{information_ratio:.3f}',
           'Your skill vs benchmark -- excess return / tracking error', _grade(information_ratio, (1.0, 0.5, 0.0)))
        _p('Treynor Ratio', f'{treynor:.3f}',
           'Return / market risk (beta) -- rewards low-beta strategies', _grade(treynor, (0.3, 0.15, 0.0)))
        _p('Omega Ratio', f'{omega:.3f}',
           'Total gains / total losses (>= 1.5 = strong edge)', _grade(omega, (1.5, 1.2, 1.0)))

        # ---- PAIN & RECOVERY ----
        print(f"  |{'':^{W}}|")
        print(f"  |  >> PAIN & RECOVERY (Drawdown-based -- 'can I stomach this?'){'':>{W-62}}|")
        print(f"  |{HR}|")
        _p('Max Drawdown', f'{max_dd*100:.2f}%',
           'Worst peak-to-trough drop (>=-20% is acceptable)', _grade(abs(max_dd*100), (10, 20, 30), reverse=True))
        _p('Calmar Ratio', f'{calmar:.3f}',
           'Annual return / max drawdown (>= 1.0 = good recovery)', _grade(calmar, (2.0, 1.0, 0.5)))
        _p('Sterling Ratio', f'{sterling:.3f}',
           'Annual return / avg of top 3 drawdowns (more stable)', _grade(sterling, (2.0, 1.0, 0.5)))
        _p('Burke Ratio', f'{burke:.3f}',
           'Penalises frequent deep drops more heavily', _grade(burke, (1.0, 0.5, 0.2)))
        _p('Ulcer Index', f'{ulcer:.2f}',
           'Depth + duration of drawdowns (lower = less stress)', _grade(ulcer, (5, 10, 20), reverse=True))

        # ---- TRADE EFFICIENCY ----
        print(f"  |{'':^{W}}|")
        print(f"  |  >> TRADE EFFICIENCY (The math of each individual trade){'':>{W-58}}|")
        print(f"  |{HR}|")
        _p('Win Rate', f'{win_rate*100:.1f}%',
           'Percentage of trades that made money (>= 50% needed)',
           _grade(win_rate*100, (55, 50, 45)))
        _p('Profit Factor', f'{pf:.3f}',
           'Gross profit / gross loss (>= 1.5 = robust edge)', _grade(pf, (1.5, 1.2, 1.0)))
        _p('Expectancy', f'{expectancy*10000:.2f} bps',
           'Expected profit per trade in basis points (> 0 = profitable)', _grade(expectancy*10000, (5, 2, 0)))
        _p('Avg Win', f'{avg_win*10000:.2f} bps',
           'Average gain on winning trades (in basis points)', '')
        _p('Avg Loss', f'{avg_loss*10000:.2f} bps',
           'Average loss on losing trades (in basis points)', '')
        _p('Tail Ratio', f'{tail:.3f}',
           'Big wins / big losses (>= 1.0 = your best days > worst)', _grade(tail, (1.2, 1.0, 0.8)))

        # ---- MARKET SKILL ----
        print(f"  |{'':^{W}}|")
        print(f"  |  >> MARKET SKILL (How well do you capture gains & dodge losses?){'':>{W-65}}|")
        print(f"  |{HR}|")
        _p('Up-Market Capture', f'{up_capture*100:.1f}%',
           'How much of bull-day gains you captured (>= 100% = full)', _grade(up_capture*100, (100, 70, 50)))
        _p('Down-Market Capture', f'{down_capture*100:.1f}%',
           'How much of bear-day pain you absorbed (<= 50% = great)', _grade(down_capture*100, (30, 50, 80), reverse=True))
        _p('Capture Ratio', f'{capture_ratio:.3f}',
           'Up / Down capture (>= 1.5 = skilful market timing)', _grade(capture_ratio, (2.0, 1.5, 1.0)))

        print(f"  |{'':^{W}}|")
        print(f"  +{HR}+")

        # Legend
        print(f"\n  * Grades: [A+] = Excellent | [B ] = Good | [C ] = Fair | [D-] = Needs Work")
        print(f"    bps = basis points (100 bps = 1%). Example: 5 bps per trade means $5 profit per $10,000 traded.")

        return result


# ═════════════════════════════════════════════════════════════════════════════
# STRATEGY DASHBOARD — 10-PANEL PLOTLY
# ═════════════════════════════════════════════════════════════════════════════

# Color palette
COLORS = {
    'bg':       '#0f172a',
    'panel':    '#1e293b',
    'grid':     '#1e293b',
    'text':     '#94a3b8',
    'cyan':     '#00d4ff',
    'green':    '#34d399',
    'red':      '#ff6b6b',
    'amber':    '#fbbf24',
    'purple':   '#a78bfa',
    'pink':     '#f472b6',
    'white':    '#e2e8f0',
}

REGIME_COLORS = {
    'BULL':   'rgba(52,211,153,0.15)',
    'PANIC':  'rgba(255,107,107,0.15)',
    'CHOPPY': 'rgba(251,191,36,0.15)',
}
REGIME_LINE = {
    'BULL':   '#34d399',
    'PANIC':  '#ff6b6b',
    'CHOPPY': '#fbbf24',
}


class StrategyDashboard:
    """Renders a premium 10-panel Plotly dashboard for any strategy."""

    def render(self, backtest_result, strategy, full_df=None):
        """
        Build and display the dashboard.

        Parameters
        ----------
        backtest_result : BacktestResult from Backtester
        strategy        : BaseStrategy instance (for metadata, importance, etc.)
        full_df         : optional full DataFrame (for regime shading over full history)
        """
        oos = backtest_result.oos_df
        meta = strategy.get_metadata()
        has_regimes = meta.get('has_regimes', False)
        forecasts = meta.get('forecasts', {})

        # Use full_df for regime plot if available, otherwise oos
        price_df = full_df if full_df is not None else oos

        fig = make_subplots(
            rows=5, cols=2,
            subplot_titles=(
                '(1) Price + Regime Shading',
                '(2) Buy/Sell Signal Overlay',
                '(3) Equity Curve: Strategy vs Buy & Hold',
                '(4) Drawdown (Underwater Plot)',
                '(5) Feature Importance (Top 15)',
                '(6) Accuracy per Regime',
                '(7) Rolling 60-Day Sharpe Ratio',
                '(8) Monthly Returns Heatmap',
                '(9) WTI Price Forecast with 95% Volatility CI',
                '(10) Performance Summary',
            ),
            vertical_spacing=0.055,
            horizontal_spacing=0.08,
            row_heights=[0.22, 0.20, 0.20, 0.20, 0.18],
            specs=[
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "table"}],
            ]
        )

        # ── Panel 1: Price + Regime Shading ──────────────────────────────
        fig.add_trace(go.Scatter(
            x=price_df.index, y=price_df['WTI_Close'], mode='lines',
            name='WTI Close', line=dict(color=COLORS['cyan'], width=1.2),
        ), row=1, col=1)

        if has_regimes and 'Regime' in price_df.columns:
            y_max = price_df['WTI_Close'].max() * 1.05
            for rname, rcolor in REGIME_COLORS.items():
                rmask = price_df['Regime'] == rname
                if rmask.any():
                    fig.add_trace(go.Scatter(
                        x=price_df.index[rmask], y=[y_max] * rmask.sum(),
                        fill='tozeroy', fillcolor=rcolor,
                        line=dict(width=0), mode='none',
                        name=f'{rname} regime', hoverinfo='skip',
                    ), row=1, col=1)

        # ── Panel 2: Signal Overlay ──────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=oos.index, y=oos['WTI_Close'], mode='lines',
            name='WTI (OOS)', line=dict(color=COLORS['text'], width=1),
            showlegend=False,
        ), row=1, col=2)

        buys = oos[oos['Signal'] == 1]
        if len(buys) > 0:
            fig.add_trace(go.Scatter(
                x=buys.index, y=buys['WTI_Close'], mode='markers',
                name='BUY', marker=dict(symbol='triangle-up', size=7, color=COLORS['green']),
            ), row=1, col=2)

        sells = oos[oos['Signal'] == -1]
        if len(sells) > 0:
            fig.add_trace(go.Scatter(
                x=sells.index, y=sells['WTI_Close'], mode='markers',
                name='SELL', marker=dict(symbol='triangle-down', size=7, color=COLORS['red']),
            ), row=1, col=2)

        # ── Panel 3: Equity Curve ────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=oos.index, y=oos['Strategy_Cumulative'], mode='lines',
            name=strategy.name, line=dict(color=COLORS['cyan'], width=2),
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=oos.index, y=oos['BnH_Cumulative'], mode='lines',
            name='Buy & Hold', line=dict(color=COLORS['text'], width=1.5, dash='dash'),
        ), row=2, col=1)
        fig.add_hline(y=1.0, line_dash='dot', line_color='rgba(255,255,255,0.3)', row=2, col=1)

        # ── Panel 4: Drawdown ────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=oos.index, y=oos['Drawdown'] * 100, mode='lines',
            fill='tozeroy', fillcolor='rgba(255,107,107,0.2)',
            line=dict(color=COLORS['red'], width=1.2),
            name='Drawdown', showlegend=False,
        ), row=2, col=2)
        fig.update_yaxes(title_text='Drawdown %', row=2, col=2)

        # ── Panel 5: Feature Importance ──────────────────────────────────
        if strategy.feature_importance is not None:
            top15 = strategy.feature_importance.head(15)
            fig.add_trace(go.Bar(
                y=top15.index[::-1], x=top15.values[::-1], orientation='h',
                marker_color=COLORS['amber'], showlegend=False,
            ), row=3, col=1)

        # ── Panel 6: Accuracy per Regime ─────────────────────────────────
        if has_regimes and 'Regime' in oos.columns and 'Prediction' in oos.columns:
            oos_valid = oos[oos['Prediction'].notna()]
            if len(oos_valid) > 0 and 'Target' in oos_valid.columns:
                from sklearn.metrics import accuracy_score
                regime_acc = oos_valid.groupby('Regime').apply(
                    lambda g: accuracy_score(g['Target'], g['Prediction'])
                    if len(g) > 0 else 0, include_groups=False
                )
                regime_n = oos_valid['Regime'].value_counts()
                fig.add_trace(go.Bar(
                    x=regime_acc.index, y=regime_acc.values,
                    marker_color=[REGIME_LINE.get(r, '#fff') for r in regime_acc.index],
                    text=[f"{v:.1%}<br>n={regime_n.get(r, 0)}"
                          for r, v in regime_acc.items()],
                    textposition='auto', showlegend=False,
                ), row=3, col=2)
                fig.add_hline(y=0.5, line_dash='dash',
                              line_color='rgba(255,107,107,0.5)',
                              annotation_text='50% (random)', row=3, col=2)

        # ── Panel 7: Rolling Sharpe ──────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=oos.index, y=oos['Rolling_Sharpe'], mode='lines',
            line=dict(color=COLORS['purple'], width=1.5), showlegend=False,
        ), row=4, col=1)
        fig.add_hline(y=0, line_dash='dash', line_color='rgba(255,255,255,0.3)', row=4, col=1)
        fig.add_hline(y=1.0, line_dash='dot', line_color='rgba(52,211,153,0.3)',
                      annotation_text='Sharpe=1.0', row=4, col=1)

        # ── Panel 8: Monthly Returns Heatmap ─────────────────────────────
        monthly = oos['Strategy_Return'].resample('ME').sum() * 100
        if len(monthly) > 0:
            monthly_df = pd.DataFrame({
                'Year': monthly.index.year,
                'Month': monthly.index.month,
                'Return': monthly.values
            })
            pivot = monthly_df.pivot_table(values='Return', index='Year',
                                           columns='Month', aggfunc='sum')
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            pivot.columns = [month_names[m - 1] for m in pivot.columns]

            fig.add_trace(go.Heatmap(
                z=pivot.values,
                x=pivot.columns.tolist(),
                y=[str(y) for y in pivot.index.tolist()],
                colorscale=[
                    [0.0, COLORS['red']],
                    [0.5, COLORS['bg']],
                    [1.0, COLORS['green']],
                ],
                zmid=0,
                text=np.round(pivot.values, 1),
                texttemplate='%{text:.1f}%',
                textfont=dict(size=10),
                showscale=False,
                hovertemplate='%{y} %{x}: %{z:.2f}%<extra></extra>',
            ), row=4, col=2)

        # ── Panel 9: WTI Price Forecast with Volatility CI ──────────────
        if forecasts:
            horizons = sorted(forecasts.keys())
            exp_prices = [forecasts[h]['expected_price'] for h in horizons]
            exp_rets   = [forecasts[h]['expected_return'] * 100 for h in horizons]
            prob_ups   = [forecasts[h]['prob_up'] for h in horizons]
            current_price = forecasts[horizons[0]].get('current_price', 0)
            labels     = [f"{h}d" for h in horizons]

            # CI bounds
            price_lows  = [forecasts[h].get('price_low', exp_prices[i]) for i, h in enumerate(horizons)]
            price_highs = [forecasts[h].get('price_high', exp_prices[i]) for i, h in enumerate(horizons)]

            # Volatility info for hover
            vol_composites = [forecasts[h].get('vol_composite', 0) for h in horizons]
            vol_garchs = [forecasts[h].get('vol_garch', 0) for h in horizons]
            vol_atrs = [forecasts[h].get('vol_atr', 0) for h in horizons]
            vol_ivs = [forecasts[h].get('vol_iv', 0) or 0 for h in horizons]

            # CI width relative to price determines bar color
            ci_widths = [(price_highs[i] - price_lows[i]) / current_price if current_price > 0 else 0
                         for i in range(len(horizons))]
            bar_colors = []
            for i, w in enumerate(ci_widths):
                if prob_ups[i] > 0.55:
                    bar_colors.append(COLORS['green'])
                elif prob_ups[i] < 0.45:
                    bar_colors.append(COLORS['red'])
                else:
                    bar_colors.append(COLORS['amber'])

            # Error bars for CI
            error_low  = [exp_prices[i] - price_lows[i] for i in range(len(horizons))]
            error_high = [price_highs[i] - exp_prices[i] for i in range(len(horizons))]

            fig.add_trace(go.Bar(
                x=labels, y=exp_prices,
                marker_color=bar_colors,
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=error_high,
                    arrayminus=error_low,
                    color=COLORS['white'],
                    thickness=1.5,
                    width=6,
                ),
                text=[f"${p:.2f}" for p in exp_prices],
                textposition='outside', showlegend=False,
                name='Expected Price',
                hovertemplate=(
                    '<b>%{x} Forecast</b><br>'
                    'Expected: $%{y:.2f}<br>'
                    'Range: $%{customdata[0]:.2f} - $%{customdata[1]:.2f}<br>'
                    'Return: %{customdata[2]:+.2f}%<br>'
                    'P(Up): %{customdata[3]:.0%}<br>'
                    'Vol: GARCH %{customdata[4]:.1f}% / ATR %{customdata[5]:.1f}% / '
                    'IV %{customdata[6]:.1f}% -> Composite %{customdata[7]:.1f}%'
                    '<extra></extra>'
                ),
                customdata=list(zip(
                    price_lows, price_highs, exp_rets, prob_ups,
                    [v*100 for v in vol_garchs], [v*100 for v in vol_atrs],
                    [v*100 for v in vol_ivs], [v*100 for v in vol_composites]
                )),
            ), row=5, col=1)

            # Composite vol line on secondary axis (overlaid)
            fig.add_trace(go.Scatter(
                x=labels, y=[v*100 for v in vol_composites],
                mode='lines+markers',
                name='Composite Vol %',
                line=dict(color=COLORS['purple'], width=2, dash='dot'),
                marker=dict(size=6, color=COLORS['purple']),
                yaxis='y9',
                showlegend=False,
                hovertemplate='%{x}: %{y:.1f}% composite vol<extra></extra>',
            ), row=5, col=1)

            # Current price reference line
            if current_price > 0:
                fig.add_hline(
                    y=current_price, line_dash='dash',
                    line_color=COLORS['amber'],
                    annotation_text=f'Current: ${current_price:.2f}',
                    annotation_font_color=COLORS['amber'],
                    row=5, col=1
                )

            fig.update_yaxes(title_text='WTI Price ($) / Composite Vol (%)', row=5, col=1)


        # ── Panel 10: Performance Summary Table ──────────────────────
        r = backtest_result
        metrics_header = ['Metric', 'Value', 'What It Means']
        metric_rows = [
            ('Total Return',       f"{r.total_return*100:+.2f}%",   'Overall profit/loss'),
            ('Buy & Hold Rtn',     f"{r.bnh_return*100:+.2f}%",     'Passive holding benchmark'),
            ('Sharpe Ratio',       f"{r.sharpe:.3f}",               'Return per unit of risk'),
            ('Sortino Ratio',      f"{r.sortino:.3f}",              'Downside-risk adjusted'),
            ('Information Ratio',  f"{r.information_ratio:.3f}",    'Skill vs benchmark'),
            ('Omega Ratio',        f"{r.omega_ratio:.3f}",          'Total gains / losses'),
            ('Max Drawdown',       f"{r.max_drawdown*100:.2f}%",    'Worst drop from peak'),
            ('Calmar Ratio',       f"{r.calmar_ratio:.3f}",         'Return / max drawdown'),
            ('Sterling Ratio',     f"{r.sterling_ratio:.3f}",       'Return / avg drawdowns'),
            ('Ulcer Index',        f"{r.ulcer_index:.2f}",          'Drawdown stress score'),
            ('Win Rate',           f"{r.win_rate*100:.1f}%",        '% of profitable trades'),
            ('Profit Factor',      f"{r.profit_factor:.3f}",        'Gross gain / gross loss'),
            ('Expectancy',         f"{r.expectancy*10000:.2f} bps", 'Avg profit per trade'),
            ('Tail Ratio',         f"{r.tail_ratio:.3f}",           'Big wins vs big losses'),
            ('Capture Ratio',      f"{r.capture_ratio:.3f}",        'Up capture / down capture'),
            ('Trading Days',       f"{r.trading_days}",             'Days in backtest'),
            ('Total Trades',       f"{r.total_trades}",             'Active trading days'),
        ]
        names = [m[0] for m in metric_rows]
        vals  = [m[1] for m in metric_rows]
        descs = [m[2] for m in metric_rows]

        # Color-code values
        value_colors = []
        for i, val_str in enumerate(vals):
            try:
                cleaned = val_str.replace('%', '').replace('+', '').replace(',', '').split()[0]
                num = float(cleaned)
                if 'Drawdown' in names[i] or 'Ulcer' in names[i]:
                    value_colors.append(COLORS['red'])
                elif num > 0:
                    value_colors.append(COLORS['green'])
                elif num < 0:
                    value_colors.append(COLORS['red'])
                else:
                    value_colors.append(COLORS['text'])
            except (ValueError, IndexError):
                value_colors.append(COLORS['text'])

        fig.add_trace(go.Table(
            header=dict(
                values=metrics_header,
                fill_color=COLORS['panel'],
                font=dict(color=COLORS['white'], size=12),
                align='left',
                line_color=COLORS['grid'],
            ),
            cells=dict(
                values=[names, vals, descs],
                fill_color=COLORS['bg'],
                font=dict(
                    color=[
                        [COLORS['white']] * len(names),
                        value_colors,
                        [COLORS['text']] * len(names),
                    ],
                    size=11,
                ),
                align='left',
                line_color=COLORS['grid'],
                height=24,
            ),
        ), row=5, col=2)

        # ── Layout ───────────────────────────────────────────────────────
        fig.update_layout(
            title=dict(
                text=(f"<b>PETROQUANT -- {strategy.name} Dashboard</b>"
                      f"<br><span style='font-size:12px;color:{COLORS['text']}'>"
                      f"Sharpe: {r.sharpe:.2f} | Return: {r.total_return*100:+.1f}% | "
                      f"MaxDD: {r.max_drawdown*100:.1f}% | "
                      f"Win Rate: {r.win_rate*100:.0f}%</span>"),
                font=dict(size=18, color='white'),
            ),
            template='plotly_dark',
            height=2000,
            width=1500,
            paper_bgcolor=COLORS['bg'],
            plot_bgcolor=COLORS['bg'],
            font=dict(color=COLORS['text']),
            hovermode='x unified',
            legend=dict(
                orientation='h', y=-0.01, x=0.5, xanchor='center',
                bgcolor='rgba(15,23,42,0.8)', bordercolor=COLORS['grid'],
            ),
        )
        fig.update_xaxes(gridcolor=COLORS['grid'], zeroline=False)
        fig.update_yaxes(gridcolor=COLORS['grid'], zeroline=False)

        # Update subplot title fonts
        for annotation in fig['layout']['annotations']:
            annotation['font'] = dict(size=13, color=COLORS['white'])

        fig.show()
        print(f"\n  [OK] Dashboard rendered for: {strategy.name}")
        return fig

    def save_html(self, fig, filepath):
        """Export dashboard to standalone HTML file."""
        fig.write_html(filepath, include_plotlyjs=True)
        print(f"  [OK] Saved: {filepath}")
