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
        print("\n  ── Backtesting ──────────────────────────────────────────")

        # Filter to out-of-sample period (where we have signals)
        has_signal = strategy_df['Probability'].notna() if 'Probability' in strategy_df.columns \
            else strategy_df['Signal'] != 0
        oos = strategy_df[has_signal].copy()

        if len(oos) == 0:
            print("    ⚠ No OOS data to backtest")
            return BacktestResult(oos_df=oos)

        # ── Returns ──────────────────────────────────────────────────────
        oos['Daily_Return']      = oos[price_col].pct_change()
        oos['Strategy_Return']   = oos['Signal'].shift(1) * oos['Daily_Return']
        oos['Strategy_Return']   = oos['Strategy_Return'].fillna(0)
        oos['BnH_Return']        = oos['Daily_Return'].fillna(0)
        oos['Strategy_Cumulative'] = (1 + oos['Strategy_Return']).cumprod()
        oos['BnH_Cumulative']      = (1 + oos['BnH_Return']).cumprod()

        # ── Drawdown ─────────────────────────────────────────────────────
        cum_max = oos['Strategy_Cumulative'].cummax()
        oos['Drawdown'] = (oos['Strategy_Cumulative'] - cum_max) / cum_max

        # ── Rolling Sharpe ───────────────────────────────────────────────
        oos['Rolling_Sharpe'] = (
            oos['Strategy_Return'].rolling(60).mean() /
            (oos['Strategy_Return'].rolling(60).std() + 1e-8)
        ) * np.sqrt(252)

        # ── Monthly Returns ──────────────────────────────────────────────
        oos['YearMonth'] = oos.index.to_period('M')

        # ── Metrics ──────────────────────────────────────────────────────
        strat_ret   = oos['Strategy_Return']
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
        )

        # ── Print Summary ────────────────────────────────────────────────
        print("  ┌─────────────────────────────────────────────────────┐")
        print("  │           STRATEGY PERFORMANCE SUMMARY              │")
        print("  ├─────────────────────────────────────────────────────┤")
        print(f"  │  Total Strategy Return       {total_ret*100:>+10.2f}%             │")
        print(f"  │  Total Buy&Hold Return       {bnh_ret*100:>+10.2f}%             │")
        print(f"  │  Annualized Sharpe           {sharpe:>10.3f}              │")
        print(f"  │  Annualized Return           {annual_ret*100:>+10.2f}%             │")
        print(f"  │  Annualized Volatility       {annual_vol*100:>10.2f}%             │")
        print(f"  │  Max Drawdown                {max_dd*100:>10.2f}%             │")
        print(f"  │  Win Rate                    {win_rate*100:>10.1f}%             │")
        print(f"  │  Profit Factor               {pf:>10.2f}              │")
        print(f"  │  Calmar Ratio                {calmar:>10.2f}              │")
        print(f"  │  Trading Days                {len(strat_ret):>10d}              │")
        print(f"  │  Total Trades                {len(trades):>10d}              │")
        print("  └─────────────────────────────────────────────────────┘")

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
                '① Price + Regime Shading',
                '② Buy/Sell Signal Overlay',
                '③ Equity Curve: Strategy vs Buy & Hold',
                '④ Drawdown (Underwater Plot)',
                '⑤ Feature Importance (Top 15)',
                '⑥ Accuracy per Regime',
                '⑦ Rolling 60-Day Sharpe Ratio',
                '⑧ Monthly Returns Heatmap',
                '⑨ Return Forecast (Multi-Horizon)',
                '⑩ Performance Summary',
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

        # ── Panel 9: Return Forecast (Multi-Horizon) ─────────────────────
        if forecasts:
            horizons = sorted(forecasts.keys())
            prob_ups = [forecasts[h]['prob_up'] for h in horizons]
            exp_rets = [forecasts[h]['expected_return'] * 100 for h in horizons]
            labels   = [f"{h}d" for h in horizons]
            bar_colors = [COLORS['green'] if p > 0.5 else COLORS['red'] for p in prob_ups]

            # Expected return bars
            fig.add_trace(go.Bar(
                x=labels, y=exp_rets,
                marker_color=bar_colors,
                text=[f"{r:+.2f}%<br>P(↑)={p:.0%}" for r, p in zip(exp_rets, prob_ups)],
                textposition='auto', showlegend=False,
                name='Expected Return',
            ), row=5, col=1)
            fig.add_hline(y=0, line_dash='dash',
                          line_color='rgba(255,255,255,0.3)', row=5, col=1)
            fig.update_yaxes(title_text='Expected Return %', row=5, col=1)

        # ── Panel 10: Performance Summary Table ──────────────────────────
        r = backtest_result
        metrics_header = ['Metric', 'Value']
        metrics_cells = [
            ['Total Return', 'Buy & Hold Return', 'Sharpe Ratio', 'Annual Return',
             'Annual Volatility', 'Max Drawdown', 'Win Rate', 'Profit Factor',
             'Calmar Ratio', 'Trading Days', 'Total Trades'],
            [f"{r.total_return*100:+.2f}%", f"{r.bnh_return*100:+.2f}%",
             f"{r.sharpe:.3f}", f"{r.annual_return*100:+.2f}%",
             f"{r.annual_vol*100:.2f}%", f"{r.max_drawdown*100:.2f}%",
             f"{r.win_rate*100:.1f}%", f"{r.profit_factor:.2f}",
             f"{r.calmar_ratio:.2f}", f"{r.trading_days}",
             f"{r.total_trades}"]
        ]

        # Color-code values
        value_colors = []
        for val_str in metrics_cells[1]:
            if '%' in val_str or val_str.replace('.', '').replace('-', '').replace('+', '').isdigit():
                try:
                    num = float(val_str.replace('%', '').replace('+', ''))
                    if 'Drawdown' in metrics_cells[0][len(value_colors)]:
                        value_colors.append(COLORS['red'])
                    elif num > 0:
                        value_colors.append(COLORS['green'])
                    elif num < 0:
                        value_colors.append(COLORS['red'])
                    else:
                        value_colors.append(COLORS['text'])
                except (ValueError, IndexError):
                    value_colors.append(COLORS['text'])
            else:
                value_colors.append(COLORS['text'])

        fig.add_trace(go.Table(
            header=dict(
                values=metrics_header,
                fill_color=COLORS['panel'],
                font=dict(color=COLORS['white'], size=13),
                align='left',
                line_color=COLORS['grid'],
            ),
            cells=dict(
                values=metrics_cells,
                fill_color=COLORS['bg'],
                font=dict(
                    color=[
                        [COLORS['white']] * len(metrics_cells[0]),
                        value_colors,
                    ],
                    size=12,
                ),
                align='left',
                line_color=COLORS['grid'],
                height=28,
            ),
        ), row=5, col=2)

        # ── Layout ───────────────────────────────────────────────────────
        fig.update_layout(
            title=dict(
                text=(f"<b>PETROQUANT — {strategy.name} Dashboard</b>"
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
        print(f"\n  ✓ Dashboard rendered for: {strategy.name}")
        return fig

    def save_html(self, fig, filepath):
        """Export dashboard to standalone HTML file."""
        fig.write_html(filepath, include_plotlyjs=True)
        print(f"  ✓ Saved: {filepath}")
