# ============================================================================
# PETROQUANT — HEALTH CHECK & DASHBOARD (fun.py)
# ============================================================================
import pandas as pd
import numpy as np
from datetime import datetime
from oil_data_pipeline import build_master_df

def run_health_check(df):
    """Checks for data staleness, gaps, and coverage."""
    print("\n" + "="*50)
    print("      PETROQUANT SYSTEM HEALTH CHECK")
    print("="*50)
    
    # 1. Staleness Check
    last_date = df.index.max()
    days_old = (datetime.now() - last_date).days
    status = "✅ CURRENT" if days_old <= 1 else "⚠ STALE"
    print(f"Latest Data Point : {last_date.date()} ({status}, {days_old} days old)")
    
    # 2. Variable Coverage (Missing Value Audit)
    print("\nVariable Coverage Audit:")
    print("-" * 30)
    for col in df.columns:
        null_count = df[col].isna().sum()
        coverage = (1 - (null_count / len(df))) * 100
        indicator = "✅" if coverage > 95 else "⚠" if coverage > 80 else "❌"
        print(f"{indicator} {col:25} | {coverage:6.1f}% coverage")

    # 3. Volatility Pulse (Fast check for 'Reflex Brain')
    if 'WTI_Close' in df.columns:
        last_return = df['WTI_Close'].pct_change().iloc[-1] * 100
        print(f"\nLast WTI Session Move: {last_return:+.2f}%")

    
    

def main():
    # Step 1: Initialize the Pipeline
    # force_refresh=False uses cache to save API limit, set to True to force update
    print("Initializing PetroQuant Pipeline...")
    master_df = build_master_df(force_refresh=False)
    print(master_df.info())

    # Step 2: Run Health Check
    run_health_check(master_df)

    # Step 3: Global Variables for Interactive Use
    # (Optional: This makes master_df available if you run 'python -i fun.py')
    return master_df

if __name__ == "__main__":
    master_df = main()