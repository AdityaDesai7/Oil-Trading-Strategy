# ============================================================================
# OIL DATA PIPELINE — SINGLE-BLOCK MASTER_DF BUILDER
# ============================================================================
# Run this ONE cell to get all 9 features ready in master_df.
# No need to run the entire notebook.
#
# USAGE (paste into a notebook cell):
#   %run oil_data_pipeline.py
#   — OR —
#   from oil_data_pipeline import build_master_df
#   master_df = build_master_df()
#
# OUTPUTS:
#   master_df  — DataFrame with 9 core features, daily frequency
#   Also saves CSV to: master_oil_features_<timestamp>.csv
# ============================================================================

import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(PIPELINE_DIR, 'data')

# Processed CSV paths (small, pipeline-friendly files)
COT_PROCESSED_CSV = os.path.join(DATA_DIR, 'cot_wti_processed.csv')
RIG_PROCESSED_CSV = os.path.join(DATA_DIR, 'rig_oil_processed.csv')

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: EIA API v2 Fetcher
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_eia(series_id, api_key):
    """Fetch weekly petroleum data from EIA API v2."""
    if not api_key:
        return None
    url = (f"https://api.eia.gov/v2/petroleum/stoc/wstk/data/"
           f"?api_key={api_key}&frequency=weekly&data[0]=value"
           f"&facets[series][]={series_id}"
           f"&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000")
    try:
        resp = requests.get(url, timeout=30)
        data = resp.json()
        if 'response' in data and 'data' in data['response']:
            df = pd.DataFrame(data['response']['data'])
            df['period'] = pd.to_datetime(df['period'])
            df = df.set_index('period').sort_index()
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            return df[['value']]
    except Exception as e:
        print(f"    ⚠ EIA fetch error ({series_id}): {e}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SMART CSV UPDATER: CFTC COT (Socrata SODA API — free, no key needed)
# ─────────────────────────────────────────────────────────────────────────────
def _update_cot_csv():
    """Check for new CFTC COT data and update the processed CSV.

    On first run: extracts WTI data from the raw 'Legacy - Futures Only.csv'
    into a small processed CSV with 3 columns: Date, Long, Short.

    On subsequent runs: queries CFTC Socrata API for data after the latest
    date in the processed CSV, appends only new rows. No duplicates.
    """
    raw_path = os.path.join(DATA_DIR, 'Legacy - Futures Only.csv')

    # --- Step 1: Create processed CSV from raw if it doesn't exist ---
    if not os.path.exists(COT_PROCESSED_CSV):
        if not os.path.exists(raw_path):
            print("    ⚠ No COT data found (no raw CSV and no processed CSV)")
            return
        print("    Creating processed COT CSV from raw data ...", end=" ", flush=True)
        try:
            raw = pd.read_csv(
                raw_path,
                usecols=['Report_Date_as_YYYY_MM_DD',
                         'Market_and_Exchange_Names',
                         'NonComm_Positions_Long_All',
                         'NonComm_Positions_Short_All'],
                low_memory=False
            )
            # Filter for WTI crude oil
            wti_mask = raw['Market_and_Exchange_Names'].str.upper().str.contains(
                'CRUDE OIL|WTI', na=False)
            wti = raw[wti_mask].copy()

            # Clean and build processed DataFrame
            wti['Date'] = pd.to_datetime(wti['Report_Date_as_YYYY_MM_DD'])
            for col in ['NonComm_Positions_Long_All', 'NonComm_Positions_Short_All']:
                wti[col] = (wti[col].astype(str)
                            .str.replace(',', '', regex=False)
                            .str.strip())
                wti[col] = pd.to_numeric(wti[col], errors='coerce')

            processed = wti[['Date', 'NonComm_Positions_Long_All',
                            'NonComm_Positions_Short_All']].copy()
            processed.columns = ['Date', 'Long', 'Short']
            processed = processed.dropna(subset=['Long', 'Short'])
            processed = processed.sort_values('Date')
            processed = processed.drop_duplicates(subset=['Date'], keep='last')
            processed.to_csv(COT_PROCESSED_CSV, index=False)
            print(f"✓ {len(processed)} rows saved")
        except Exception as e:
            print(f"✗ FAILED: {e}")
            return

    # --- Step 2: Find latest date in processed CSV ---
    try:
        existing = pd.read_csv(COT_PROCESSED_CSV)
        existing['Date'] = pd.to_datetime(existing['Date'])
        latest_date = existing['Date'].max()
    except Exception as e:
        print(f"    ⚠ Could not read processed COT CSV: {e}")
        return

    # --- Step 3: Query Socrata API for data AFTER latest_date ---
    fetch_from = latest_date + timedelta(days=1)
    fetch_to = datetime.now()

    base_url = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
    params = {
        '$where': (
            f"cftc_contract_market_code='067651'"
            f" AND report_date_as_yyyy_mm_dd >= '{fetch_from:%Y-%m-%dT00:00:00.000}'"
            f" AND report_date_as_yyyy_mm_dd <= '{fetch_to:%Y-%m-%dT00:00:00.000}'"
        ),
        '$select': 'report_date_as_yyyy_mm_dd,noncomm_positions_long_all,noncomm_positions_short_all',
        '$order': 'report_date_as_yyyy_mm_dd ASC',
        '$limit': 5000,
    }
    try:
        resp = requests.get(base_url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            print(f"    ✓ COT data is up-to-date (latest: {latest_date.date()})")
            return

        # Build new rows in processed CSV format
        new_rows = []
        for record in data:
            dt = record.get('report_date_as_yyyy_mm_dd', '')[:10]
            if pd.to_datetime(dt) > latest_date:
                new_rows.append({
                    'Date': dt,
                    'Long': record.get('noncomm_positions_long_all', ''),
                    'Short': record.get('noncomm_positions_short_all', ''),
                })

        if not new_rows:
            print(f"    ✓ COT data is up-to-date (latest: {latest_date.date()})")
            return

        # Append to processed CSV
        new_df = pd.DataFrame(new_rows)
        new_df.to_csv(COT_PROCESSED_CSV, mode='a', header=False, index=False)
        new_max = pd.to_datetime(new_df['Date']).max()
        print(f"    ✓ Appended {len(new_df)} new COT rows (up to {new_max.date()})")

    except Exception as e:
        print(f"    ⚠ COT update error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# SMART CSV UPDATER: Baker Hughes Rig Count
# ─────────────────────────────────────────────────────────────────────────────
def _update_rig_csv():
    """Check for new Baker Hughes rig count data and update the processed CSV.

    On first run: extracts US Oil rig data from the raw 'Rig Counts.csv'
    into a small processed CSV with 2 columns: Date, US_Oil_Rigs.

    On subsequent runs: downloads latest data from Baker Hughes website,
    appends only new rows. No duplicates.
    """
    raw_path = os.path.join(DATA_DIR, 'Rig Counts.csv')

    # --- Step 1: Create processed CSV from raw if it doesn't exist ---
    if not os.path.exists(RIG_PROCESSED_CSV):
        if not os.path.exists(raw_path):
            print("    ⚠ No rig data found (no raw CSV and no processed CSV)")
            return
        print("    Creating processed Rig CSV from raw data ...", end=" ", flush=True)
        try:
            rig_raw = pd.read_csv(raw_path)
            rig_raw['US_PublishDate'] = pd.to_datetime(
                rig_raw['US_PublishDate'], dayfirst=True)

            # Group by date and drill type, get Oil rigs
            rig_summary = (rig_raw
                          .groupby(['US_PublishDate', 'DrillFor'])['Rig Count Value']
                          .sum()
                          .unstack(fill_value=0))

            oil_col = None
            for candidate in rig_summary.columns:
                if str(candidate).strip().upper() == 'OIL':
                    oil_col = candidate
                    break

            if oil_col is not None:
                processed = pd.DataFrame({
                    'Date': rig_summary.index,
                    'US_Oil_Rigs': rig_summary[oil_col].values
                })
            else:
                processed = pd.DataFrame({
                    'Date': rig_summary.index,
                    'US_Oil_Rigs': rig_summary.iloc[:, 0].values
                })

            processed = processed.sort_values('Date')
            processed = processed.drop_duplicates(subset=['Date'], keep='last')
            processed.to_csv(RIG_PROCESSED_CSV, index=False)
            print(f"✓ {len(processed)} rows saved")
        except Exception as e:
            print(f"✗ FAILED: {e}")
            return

    # --- Step 2: Find latest date in processed CSV ---
    try:
        existing = pd.read_csv(RIG_PROCESSED_CSV)
        existing['Date'] = pd.to_datetime(existing['Date'])
        latest_date = existing['Date'].max()
    except Exception as e:
        print(f"    ⚠ Could not read processed Rig CSV: {e}")
        return

    # --- Step 3: Try to download new data from Baker Hughes ---
    BH_URLS = [
        "https://rigcount.bakerhughes.com/static-files/5bae3541-2f5a-4401-a2ed-92560906dfbd",
    ]

    for url in BH_URLS:
        try:
            print(f"    Downloading Baker Hughes data ...", end=" ", flush=True)
            resp = requests.get(url, timeout=60, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            resp.raise_for_status()

            tmp_path = os.path.join(PIPELINE_DIR, '_tmp_bh_rigcount.xlsx')
            with open(tmp_path, 'wb') as f:
                f.write(resp.content)

            # Try reading the "Current" sheet
            try:
                excel_df = pd.read_excel(tmp_path, sheet_name='Current', header=5)
            except Exception:
                try:
                    excel_df = pd.read_excel(tmp_path, header=5)
                except Exception:
                    excel_df = None

            # Clean up temp file
            try:
                os.remove(tmp_path)
            except OSError:
                pass

            if excel_df is None or len(excel_df) == 0:
                print("⚠ empty data")
                continue

            # Find date column
            date_col = None
            for candidate in excel_df.columns:
                cstr = str(candidate).lower()
                if 'date' in cstr or 'publish' in cstr:
                    date_col = candidate
                    break
            if date_col is None:
                date_col = excel_df.columns[0]

            excel_df[date_col] = pd.to_datetime(excel_df[date_col], errors='coerce')
            excel_df = excel_df.dropna(subset=[date_col])

            # Find oil rig count column
            oil_count_col = None
            for candidate in excel_df.columns:
                cstr = str(candidate).lower()
                if 'oil' in cstr:
                    oil_count_col = candidate
                    break

            if oil_count_col is None:
                print("⚠ no 'Oil' column found in downloaded data")
                continue

            # Filter for rows AFTER latest_date
            new_data = excel_df[excel_df[date_col] > latest_date].copy()

            if len(new_data) == 0:
                print(f"✓ Rig data is up-to-date (latest: {latest_date.date()})")
                return

            # Append to processed CSV
            new_rows = pd.DataFrame({
                'Date': new_data[date_col].dt.strftime('%Y-%m-%d'),
                'US_Oil_Rigs': pd.to_numeric(new_data[oil_count_col], errors='coerce')
            })
            new_rows = new_rows.dropna(subset=['US_Oil_Rigs'])
            new_rows.to_csv(RIG_PROCESSED_CSV, mode='a', header=False, index=False)
            new_max_dt = pd.to_datetime(new_rows['Date']).max()
            print(f"✓ Appended {len(new_rows)} new rig rows (up to {new_max_dt.date()})")
            return

        except Exception as e:
            print(f"⚠ download failed: {e}")
            continue

    print(f"    ⚠ Could not download new rig data, using existing CSV (latest: {latest_date.date()})")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════
def build_master_df(years=5, force_refresh=False, save_csv=True):
    """
    Build master_df with all 9 core features in a single call.

    Parameters
    ----------
    years         : int, years of history to fetch (default 5)
    force_refresh : bool, if True always re-fetch from APIs.
                    if False, loads the latest cached CSV if < 24h old.
    save_csv      : bool, save a timestamped CSV after building.

    Returns
    -------
    master_df : pd.DataFrame with DatetimeIndex and 9+ columns
    """
    END_DATE   = datetime.now()
    START_DATE = END_DATE - timedelta(days=365 * years)

    # ── Check for cached CSV (skip API calls if recent) ──────────────────
    if not force_refresh:
        cached = _find_latest_cache(PIPELINE_DIR)
        if cached:
            print(f"✓ Loading cached data: {os.path.basename(cached)}")
            df = pd.read_csv(cached, index_col=0, parse_dates=True)
            print(f"  {df.shape[0]} rows × {df.shape[1]} cols | "
                  f"{df.index.min().date()} → {df.index.max().date()}")
            return df

    print("=" * 70)
    print("  OIL DATA PIPELINE — Fetching All 9 Variables")
    print("=" * 70)
    print(f"  Date range: {START_DATE:%Y-%m-%d} → {END_DATE:%Y-%m-%d}")
    print()

    # ── Load API Keys ────────────────────────────────────────────────────
    load_dotenv(os.path.join(PIPELINE_DIR, '.env'))
    FRED_KEY = os.getenv('FRED_API_KEY')
    EIA_KEY  = os.getenv('EIA_API_KEY')
    fred     = Fred(api_key=FRED_KEY) if FRED_KEY else None

    print(f"  FRED API: {'✓ Ready' if FRED_KEY else '✗ Missing key'}")
    print(f"  EIA  API: {'✓ Ready' if EIA_KEY else '✗ Missing key'}")
    print()

    # ── Smart-update CSVs before reading them ────────────────────────────
    print("── Updating local CSVs (if new Friday data available) " + "─" * 18)
    _update_cot_csv()
    _update_rig_csv()
    print()

    # =====================================================================
    # BAND 1: FAST VARIABLES (Daily)
    # =====================================================================
    print("── FAST Variables (Daily) " + "─" * 46)

    # 1. WTI Crude Oil (CL=F)
    print("  [1/9] WTI_Close  ← Yahoo Finance CL=F ...", end=" ")
    wti_raw = yf.download('CL=F', start=START_DATE, end=END_DATE, progress=False)
    wti_df = wti_raw[['Close']].copy()
    wti_df.columns = ['WTI_Close']
    if isinstance(wti_df.columns, pd.MultiIndex):
        wti_df.columns = wti_df.columns.get_level_values(0)
    print(f"✓ {len(wti_df)} rows")

    # 2. Brent Crude Oil (BZ=F)
    print("  [2/9] Brent_Close ← Yahoo Finance BZ=F ...", end=" ")
    brent_raw = yf.download('BZ=F', start=START_DATE, end=END_DATE, progress=False)
    brent_df = brent_raw[['Close']].copy()
    brent_df.columns = ['Brent_Close']
    if isinstance(brent_df.columns, pd.MultiIndex):
        brent_df.columns = brent_df.columns.get_level_values(0)
    print(f"✓ {len(brent_df)} rows")

    # 3. OVX — Oil Volatility Index (FRED)
    print("  [3/9] OVX        ← FRED OVXCLS ...", end=" ")
    if fred:
        try:
            ovx_data = fred.get_series('OVXCLS',
                                       observation_start=START_DATE,
                                       observation_end=END_DATE)
            ovx_df = pd.DataFrame({'OVX': ovx_data})
            print(f"✓ {len(ovx_df)} rows")
        except Exception as e:
            print(f"⚠ Error: {e}")
            ovx_df = pd.DataFrame({'OVX': []})
    else:
        print("⚠ Skipped (no FRED key)")
        ovx_df = pd.DataFrame({'OVX': []})

    # 4. USD Index (FRED)
    print("  [4/9] USD_Index  ← FRED DTWEXBGS ...", end=" ")
    if fred:
        try:
            dxy_data = fred.get_series('DTWEXBGS',
                                       observation_start=START_DATE,
                                       observation_end=END_DATE)
            dxy_df = pd.DataFrame({'USD_Index': dxy_data})
            print(f"✓ {len(dxy_df)} rows")
        except Exception as e:
            print(f"⚠ Error: {e}")
            dxy_df = pd.DataFrame({'USD_Index': []})
    else:
        print("⚠ Skipped (no FRED key)")
        dxy_df = pd.DataFrame({'USD_Index': []})

    # Combine FAST into base DataFrame (daily index)
    fast_df = pd.concat([wti_df, brent_df, ovx_df, dxy_df], axis=1)
    fast_df = fast_df.dropna(subset=['WTI_Close', 'Brent_Close'])
    fast_df.index = pd.to_datetime(fast_df.index)
    if fast_df.index.tz is not None:
        fast_df.index = fast_df.index.tz_localize(None)

    print()

    # =====================================================================
    # BAND 2: MEDIUM VARIABLES (Weekly → resampled to Daily)
    # =====================================================================
    print("── MEDIUM Variables (Weekly) " + "─" * 42)

    # 5. Crack Spread 3-2-1 (calculated from RBOB + Heating Oil)
    print("  [5/9] Crack_3_2_1 ← Yahoo Finance RB=F, HO=F ...", end=" ")
    try:
        rbob_raw = yf.download('RB=F', start=START_DATE, end=END_DATE, progress=False)
        ho_raw   = yf.download('HO=F', start=START_DATE, end=END_DATE, progress=False)

        rbob_close = rbob_raw['Close'].squeeze() * 42
        ho_close   = ho_raw['Close'].squeeze() * 42
        wti_close  = wti_raw['Close'].squeeze()

        crack_df = pd.DataFrame({
            'Crack_3_2_1': (2 * rbob_close + ho_close - 3 * wti_close) / 3
        })
        if crack_df.index.tz is not None:
            crack_df.index = crack_df.index.tz_localize(None)
        print(f"✓ {crack_df['Crack_3_2_1'].notna().sum()} rows")
    except Exception as e:
        print(f"⚠ Error: {e}")
        crack_df = pd.DataFrame({'Crack_3_2_1': []})

    # 6. Net Speculative Position (from processed COT CSV, auto-updated)
    print("  [6/9] Net_Speculative_Position ← processed COT CSV ...", end=" ", flush=True)
    cot_df = pd.DataFrame({'Net_Speculative_Position': []})

    if os.path.exists(COT_PROCESSED_CSV):
        try:
            cot_raw = pd.read_csv(COT_PROCESSED_CSV)
            cot_raw['Date'] = pd.to_datetime(cot_raw['Date'])
            cot_raw.set_index('Date', inplace=True)
            cot_raw.sort_index(inplace=True)
            cot_raw = cot_raw.loc[START_DATE:END_DATE]

            for col in ['Long', 'Short']:
                cot_raw[col] = pd.to_numeric(cot_raw[col], errors='coerce')

            cot_df = pd.DataFrame({
                'Net_Speculative_Position': cot_raw['Long'] - cot_raw['Short']
            })

            if cot_df.index.duplicated().any():
                cot_df = cot_df.groupby(cot_df.index).mean()

            print(f"✓ {len(cot_df)} weeks")
        except Exception as e:
            print(f"✗ FAILED: {e}")
    else:
        print("✗ No processed COT CSV found")

    # 7. US Crude Oil Stocks (EIA API)
    print("  [7/9] Crude_Stocks_1000bbl ← EIA WCESTUS1 ...", end=" ")
    eia_crude = _fetch_eia('WCESTUS1', EIA_KEY)
    if eia_crude is not None:
        eia_crude.columns = ['Crude_Stocks_1000bbl']
        print(f"✓ {len(eia_crude)} weeks")
    else:
        print("⚠ Skipped (no EIA key or error)")
        eia_crude = pd.DataFrame({'Crude_Stocks_1000bbl': []})

    # 8. US Oil Rig Count (from processed Rig CSV, auto-updated)
    print("  [8/9] US_Oil_Rigs ← processed Rig CSV ...", end=" ", flush=True)
    rig_df = pd.DataFrame({'US_Oil_Rigs': []})

    if os.path.exists(RIG_PROCESSED_CSV):
        try:
            rig_raw = pd.read_csv(RIG_PROCESSED_CSV)
            rig_raw['Date'] = pd.to_datetime(rig_raw['Date'])
            rig_raw.set_index('Date', inplace=True)
            rig_raw.sort_index(inplace=True)
            rig_raw = rig_raw.loc[START_DATE:END_DATE]

            rig_df = pd.DataFrame({
                'US_Oil_Rigs': pd.to_numeric(rig_raw['US_Oil_Rigs'], errors='coerce')
            })

            if rig_df.index.duplicated().any():
                rig_df = rig_df.groupby(rig_df.index).mean()

            print(f"✓ {len(rig_df)} weeks")
        except Exception as e:
            print(f"✗ FAILED: {e}")
    else:
        print("✗ No processed Rig CSV found")

    print()

    # =====================================================================
    # BAND 3: SLOW VARIABLES (Monthly → resampled to Daily)
    # =====================================================================
    print("── SLOW Variables (Monthly) " + "─" * 43)

    # 9. Strategic Petroleum Reserve (EIA API)
    print("  [9/9] SPR_Stocks_1000bbl ← EIA WCSSTUS1 ...", end=" ")
    eia_spr = _fetch_eia('WCSSTUS1', EIA_KEY)
    if eia_spr is not None:
        eia_spr.columns = ['SPR_Stocks_1000bbl']
        print(f"✓ {len(eia_spr)} weeks")
    else:
        print("⚠ Skipped (no EIA key or error)")
        eia_spr = pd.DataFrame({'SPR_Stocks_1000bbl': []})

    print()

    # =====================================================================
    # CONSOLIDATION — Merge everything onto the daily index
    # =====================================================================
    print("── Consolidating into master_df " + "─" * 39)

    master_df = fast_df.copy()

    # Helper: clean, deduplicate, resample weekly→daily, join
    def _join_weekly(master, weekly_df, col_name):
        if weekly_df.empty or col_name not in weekly_df.columns:
            print(f"    ⚠ Skipping {col_name} (empty or missing column)")
            return master
        clean = weekly_df[[col_name]].copy()
        clean.index = pd.to_datetime(clean.index)
        if clean.index.tz is not None:
            clean.index = clean.index.tz_localize(None)
        if clean.index.duplicated().any():
            clean = clean[~clean.index.duplicated(keep='last')]
        daily = clean.resample('D').ffill()
        master = master.join(daily, how='left')
        nn = master[col_name].notna().sum()
        print(f"    ✓ Joined {col_name}: {nn} non-null in master_df")
        return master

    master_df = _join_weekly(master_df, crack_df, 'Crack_3_2_1')
    master_df = _join_weekly(master_df, cot_df, 'Net_Speculative_Position')
    master_df = _join_weekly(master_df, eia_crude, 'Crude_Stocks_1000bbl')
    master_df = _join_weekly(master_df, rig_df, 'US_Oil_Rigs')
    master_df = _join_weekly(master_df, eia_spr, 'SPR_Stocks_1000bbl')

    # ── Forward-fill rules ───────────────────────────────────────────────
    ffill_rules = {
        'OVX': 3,
        'USD_Index': 3,
        'Crack_3_2_1': 7,
        'Net_Speculative_Position': 7,
        'Crude_Stocks_1000bbl': 7,
        'US_Oil_Rigs': 7,
        'SPR_Stocks_1000bbl': 30,
    }
    for col, limit in ffill_rules.items():
        if col in master_df.columns:
            master_df[col] = master_df[col].ffill(limit=limit)

    # Drop rows without WTI price
    master_df = master_df.dropna(subset=['WTI_Close'])

    # ── Status Report ────────────────────────────────────────────────────
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

    expected_9 = ['WTI_Close', 'Brent_Close', 'OVX', 'USD_Index',
                  'Crack_3_2_1', 'Net_Speculative_Position',
                  'Crude_Stocks_1000bbl', 'US_Oil_Rigs', 'SPR_Stocks_1000bbl']
    missing_vars = []

    for col in master_df.columns:
        nn = master_df[col].notna().sum()
        pct = nn / len(master_df) * 100
        bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
        print(f"  {col:30s} {nn:5d}    {bar} {pct:.0f}%")

    for v in expected_9:
        if v not in master_df.columns:
            missing_vars.append(v)
            print(f"  ✗ {v:30s}  MISSING — not in master_df!")

    if missing_vars:
        print(f"\n  ⚠ WARNING: {len(missing_vars)} variable(s) missing: {missing_vars}")
    else:
        print(f"\n  ✓ All 9 core variables present!")

    # ── Save CSV ─────────────────────────────────────────────────────────
    if save_csv:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(PIPELINE_DIR,
                                f'master_oil_features_{ts}.csv')
        master_df.to_csv(csv_path)
        print(f"\n  ✓ Saved: {os.path.basename(csv_path)}")

    print("=" * 70)
    return master_df


def _find_latest_cache(directory, max_age_hours=24):
    """Find the most recent master_oil_features CSV if < max_age_hours old."""
    import glob
    pattern = os.path.join(directory, 'master_oil_features_*.csv')
    files = glob.glob(pattern)
    if not files:
        return None
    latest = max(files, key=os.path.getmtime)
    age_hours = (datetime.now().timestamp() - os.path.getmtime(latest)) / 3600
    if age_hours < max_age_hours:
        return latest
    return None


# ═════════════════════════════════════════════════════════════════════════════
# AUTO-RUN when executed with %run or python oil_data_pipeline.py
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__' or '__IPYTHON__' in dir():
    master_df = build_master_df(force_refresh=True)
    print(f"\n✓ master_df is ready to use ({len(master_df)} rows)")