# ============================================================================
# PETROQUANT — STRATEGY RUNNER
# ============================================================================
# Single entry point to run any strategy end-to-end:
#   Data Pipeline -> Strategy -> Backtest -> Dashboard
#
# USAGE:
#   python run_strategy.py
#
# To run a different timeframe, change fwd_days below.
# To add a new strategy, import it and add to STRATEGIES dict.
# ============================================================================

import os
import sys

# Force UTF-8 output so box-drawing characters work on Windows terminals
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# Ensure we can import from the project directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from oil_data_pipeline_new import build_master_df
from strategy import HMMXGBoostStrategy
from dashboard import Backtester, StrategyDashboard


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
STRATEGIES = {
    'hmm_xgb': {
        'class': HMMXGBoostStrategy,
        'params': {'fwd_days': 5},
        'description': 'HMM + XGBoost Regime-Aware Strategy',
    },
}

# Which strategies to run (set to None to run all)
RUN_STRATEGIES = None  # or ['hmm_xgb']


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Step 1: Load data from pipeline ──────────────────────────────────
    print("\n" + "=" * 70)
    print("  PETROQUANT -- STRATEGY RUNNER")
    print("=" * 70)
    print("  Loading data from pipeline...")
    print("  (Fetching oil prices, volatility, inventories, and macro data)\n")

    master_df = build_master_df(force_refresh=False)

    if master_df is None or len(master_df) == 0:
        print("  [FAIL] Failed to load data. Run oil_data_pipeline_new.py first.")
        return

    print(f"\n  [OK] Loaded {len(master_df)} rows x {len(master_df.columns)} cols")
    print(f"  [OK] Range: {master_df.index.min().date()} -> {master_df.index.max().date()}")

    # ── Step 2: Run selected strategies ──────────────────────────────────
    to_run = RUN_STRATEGIES or list(STRATEGIES.keys())

    for key in to_run:
        if key not in STRATEGIES:
            print(f"\n  [WARN] Unknown strategy: {key}")
            continue

        config = STRATEGIES[key]
        print(f"\n{'='*70}")
        print(f"  Strategy: {config['description']}")
        print(f"{'='*70}")

        # Initialize strategy
        strategy = config['class'](**config['params'])

        # Run strategy pipeline
        result_df = strategy.run(master_df)

        # Backtest
        bt = Backtester()
        bt_result = bt.run(result_df)

        # Dashboard
        dash = StrategyDashboard()
        fig = dash.render(bt_result, strategy, full_df=result_df)

        # Save HTML
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
        os.makedirs(output_dir, exist_ok=True)
        html_path = os.path.join(output_dir, f'dashboard_{key}.html')
        dash.save_html(fig, html_path)

    print(f"\n{'='*70}")
    print("  [OK] ALL STRATEGIES COMPLETE")
    print("    Check the /output/ folder for your interactive HTML dashboard!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
