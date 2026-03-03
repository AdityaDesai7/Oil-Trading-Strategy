# ============================================================================
# OIL DATA PIPELINE — MODULAR MASTER_DF BUILDER
# ============================================================================
# Fetches all features defined in features.py and consolidates them into
# a single master_df with daily frequency.
#
# USAGE:
#   %run oil_data_pipeline.py
#   — OR —
#   from oil_data_pipeline import build_master_df
#   master_df = build_master_df()
#
# TO ADD A NEW FEATURE:
#   Edit features.py — add a fetch function + append to FEATURES list.
#   No changes needed here.
# ============================================================================

import pandas as pd
import os
import glob
from datetime import datetime, timedelta
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

from features import FEATURES

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(PIPELINE_DIR, 'data')
OUTPUT_DIR   = os.path.join(PIPELINE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _find_latest_cache(directory, max_age_hours=24):
    """Find the most recent master_oil_features CSV if < max_age_hours old."""
    pattern = os.path.join(directory, 'master_oil_features_*.csv')
    files = glob.glob(pattern)
    if not files:
        return None
    latest = max(files, key=os.path.getmtime)
    age_hours = (datetime.now().timestamp() - os.path.getmtime(latest)) / 3600
    if age_hours < max_age_hours:
        return latest
    return None


def _join_feature(master, feat_df, col_name, ffill_limit):
    """Clean, deduplicate, resample to daily, and join a feature onto master."""
    if feat_df.empty or col_name not in feat_df.columns:
        print(f"    ⚠ Skipping {col_name} (empty or missing)")
        return master

    clean = feat_df[[col_name]].copy()
    clean.index = pd.to_datetime(clean.index)
    if clean.index.tz is not None:
        clean.index = clean.index.tz_localize(None)
    if clean.index.duplicated().any():
        clean = clean[~clean.index.duplicated(keep='last')]

    # Resample to daily and forward-fill
    daily = clean.resample('D').ffill()
    master = master.join(daily, how='left')

    # Apply bounded forward-fill
    if ffill_limit > 0 and col_name in master.columns:
        master[col_name] = master[col_name].ffill(limit=ffill_limit)

    nn = master[col_name].notna().sum()
    print(f"    ✓ {col_name}: {nn} non-null")
    return master


# ═════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════
def build_master_df(years=5, force_refresh=False, save_csv=True):
    """
    Build master_df from all features in the FEATURES registry.

    Parameters
    ----------
    years         : int, years of history (default 5)
    force_refresh : bool, if True always re-fetch from APIs
    save_csv      : bool, save a timestamped CSV after building

    Returns
    -------
    master_df : pd.DataFrame with DatetimeIndex
    """
    # END_DATE   = datetime.now()
    END_DATE=datetime(2025,12,31)
    START_DATE = END_DATE - timedelta(days=365 * years)

    # ── Check cache ──────────────────────────────────────────────────────
    if not force_refresh:
        cached = _find_latest_cache(OUTPUT_DIR)
        if cached:
            print(f"✓ Loading cached: {os.path.basename(cached)}")
            df = pd.read_csv(cached, index_col=0, parse_dates=True)
            print(f"  {df.shape[0]} rows × {df.shape[1]} cols | "
                  f"{df.index.min().date()} → {df.index.max().date()}")
            return df

    # ── Load config ──────────────────────────────────────────────────────
    load_dotenv(os.path.join(PIPELINE_DIR, '.env'))
    config = {
        'FRED_API_KEY': os.getenv('FRED_API_KEY'),
        'EIA_API_KEY' : os.getenv('EIA_API_KEY'),
        'DATA_DIR'    : DATA_DIR,
        'PIPELINE_DIR': PIPELINE_DIR,
    }

    total = len(FEATURES)
    print("=" * 70)
    print(f"  OIL DATA PIPELINE — Fetching {total} Features")
    print("=" * 70)
    print(f"  Range: {START_DATE:%Y-%m-%d} → {END_DATE:%Y-%m-%d}")
    print(f"  FRED : {'✓' if config['FRED_API_KEY'] else '✗'} | "
          f"EIA: {'✓' if config['EIA_API_KEY'] else '✗'}")
    print()

    # ── Fetch all features ───────────────────────────────────────────────
    fetched = {}            # name → DataFrame
    base_frames = []        # DataFrames for base index
    current_band = None

    for i, feat in enumerate(FEATURES, 1):
        name   = feat['name']
        band   = feat['band']
        source = feat.get('source', '')

        # Print band header
        if band != current_band:
            labels = {'fast': 'FAST (Daily)', 'medium': 'MEDIUM (Weekly/Monthly)',
                      'slow': 'SLOW (Monthly)'}
            print(f"── {labels.get(band, band.upper())} " + "─" * 50)
            current_band = band

        print(f"  [{i}/{total}] {name:30s} ← {source} ...", end=" ", flush=True)

        try:
            df = feat['fetch'](START_DATE, END_DATE, config)
            rows = len(df) if not df.empty else 0
            nn = df[name].notna().sum() if name in df.columns and not df.empty else 0
            print(f"✓ {nn} rows")
            fetched[name] = (df, feat)

            if feat.get('is_base'):
                base_frames.append(df)
        except Exception as e:
            print(f"✗ {e}")
            fetched[name] = (pd.DataFrame({name: []}), feat)

    print()

    # ── Build base index from is_base features ───────────────────────────
    print("── Consolidating " + "─" * 53)

    if not base_frames:
        raise RuntimeError("No base features loaded — cannot build master_df")

    master_df = pd.concat(base_frames, axis=1)
    # Keep only rows where ALL base features have data
    base_names = [f['name'] for f in FEATURES if f.get('is_base')]
    master_df = master_df.dropna(subset=base_names)
    master_df.index = pd.to_datetime(master_df.index)
    if master_df.index.tz is not None:
        master_df.index = master_df.index.tz_localize(None)

    # ── Join non-base features ───────────────────────────────────────────
    for name, (df, feat) in fetched.items():
        if feat.get('is_base'):
            continue  # already in master_df
        master_df = _join_feature(master_df, df, name, feat['ffill_limit'])

    # Drop rows without WTI price
    if 'WTI_Close' in master_df.columns:
        master_df = master_df.dropna(subset=['WTI_Close'])

    # ── Status report ────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  PIPELINE COMPLETE — master_df READY")
    print("=" * 70)
    print(f"  Rows:  {len(master_df)}")
    print(f"  Cols:  {len(master_df.columns)}")
    print(f"  Range: {master_df.index.min().date()} → {master_df.index.max().date()}")
    print()
    print("  Column                        Non-Null   Coverage")
    print("  " + "─" * 55)

    expected = [f['name'] for f in FEATURES]
    missing  = []

    for col in master_df.columns:
        nn  = master_df[col].notna().sum()
        pct = nn / len(master_df) * 100
        bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
        print(f"  {col:30s} {nn:5d}    {bar} {pct:.0f}%")

    for v in expected:
        if v not in master_df.columns:
            missing.append(v)
            print(f"  ✗ {v:30s}  MISSING!")

    if missing:
        print(f"\n  ⚠ {len(missing)} variable(s) missing: {missing}")
    else:
        print(f"\n  ✓ All {len(expected)} features present!")

    # ── Save CSV ─────────────────────────────────────────────────────────
    if save_csv:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(OUTPUT_DIR, f'master_oil_features_{ts}.csv')
        master_df.to_csv(csv_path)
        print(f"\n  ✓ Saved: {os.path.basename(csv_path)}")

    print("=" * 70)
    return master_df


# ═════════════════════════════════════════════════════════════════════════════
# AUTO-RUN
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__' or '__IPYTHON__' in dir():
    master_df = build_master_df(force_refresh=True)
    print(f"\n✓ master_df is ready ({len(master_df)} rows)")
