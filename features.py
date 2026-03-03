# ============================================================================
# FEATURES REGISTRY — All Oil Market Features in One Place
# ============================================================================
# Each feature is defined as a dict with:
#   name       : column name in master_df
#   band       : "fast" | "medium" | "slow"
#   frequency  : "daily" | "weekly" | "monthly"
#   ffill_limit: max days to forward-fill (0 = no extra ffill)
#   is_base    : if True, used to build the daily base index
#   fetch      : callable(start, end, config) → pd.DataFrame with 1 column
#
# TO ADD A NEW FEATURE:
#   1. Write a fetch_xxx(start, end, config) function below
#   2. Append a dict to the FEATURES list at the bottom
#   That's it — the pipeline will pick it up automatically.
# ============================================================================

import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import requests
import os
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# FETCH FUNCTIONS — one per feature
# ─────────────────────────────────────────────────────────────────────────────

# ── BAND 1: FAST (Daily) ────────────────────────────────────────────────────

def fetch_wti(start, end, config):
    """WTI Crude Oil price from Yahoo Finance (CL=F)."""
    raw = yf.download('CL=F', start=start, end=end, progress=False)
    df = raw[['Close']].copy()
    df.columns = ['WTI_Close']
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def fetch_brent(start, end, config):
    """Brent Crude Oil price from Yahoo Finance (BZ=F)."""
    raw = yf.download('BZ=F', start=start, end=end, progress=False)
    df = raw[['Close']].copy()
    df.columns = ['Brent_Close']
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def fetch_ovx(start, end, config):
    """Oil Volatility Index from FRED (OVXCLS)."""
    fred_key = config.get('FRED_API_KEY')
    if not fred_key:
        return pd.DataFrame({'OVX': []})
    try:
        fred = Fred(api_key=fred_key)
        data = fred.get_series('OVXCLS',
                               observation_start=start,
                               observation_end=end)
        return pd.DataFrame({'OVX': data})
    except Exception as e:
        print(f"⚠ OVX error: {e}")
        return pd.DataFrame({'OVX': []})


def fetch_usd_index(start, end, config):
    """USD Index (DXY) from Yahoo Finance (DX-Y.NYB)."""
    try:
        ticker = yf.Ticker("DX-Y.NYB")
        raw = ticker.history(
            start=start.strftime('%Y-%m-%d') if hasattr(start, 'strftime') else start,
            end=end.strftime('%Y-%m-%d') if hasattr(end, 'strftime') else end,
            interval="1d"
        )
        if raw.empty:
            return pd.DataFrame({'USD_Index': []})
        df = pd.DataFrame({'USD_Index': raw['Close']})
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = 'Date'
        return df
    except Exception as e:
        print(f"⚠ USD_Index error: {e}")
        return pd.DataFrame({'USD_Index': []})


# ── BAND 2: MEDIUM (Weekly / Monthly → resampled) ──────────────────────────

def fetch_crack_spread(start, end, config):
    """3-2-1 Crack Spread from Yahoo Finance (RB=F, HO=F, CL=F)."""
    try:
        rbob = yf.download('RB=F', start=start, end=end, progress=False)
        ho   = yf.download('HO=F', start=start, end=end, progress=False)
        wti  = yf.download('CL=F', start=start, end=end, progress=False)

        rbob_close = rbob['Close'].squeeze() * 42
        ho_close   = ho['Close'].squeeze() * 42
        wti_close  = wti['Close'].squeeze()

        df = pd.DataFrame({
            'Crack_3_2_1': (2 * rbob_close + ho_close - 3 * wti_close) / 3
        })
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    except Exception as e:
        print(f"⚠ Crack_3_2_1 error: {e}")
        return pd.DataFrame({'Crack_3_2_1': []})


def fetch_cot_net_spec(start, end, config):
    """
    CFTC COT Net Speculative Position via Socrata Open Data API.
    No API key needed. Searches for CRUDE OIL instrument.
    """
    try:
        url = "https://publicreporting.cftc.gov/resource/jun7-fc8e.json"
        instrument = "CRUDE OIL"

        start_fmt = pd.to_datetime(start).strftime("%Y-%m-%dT00:00:00.000")
        end_fmt   = pd.to_datetime(end).strftime("%Y-%m-%dT00:00:00.000")

        params = {
            "$where": (f"market_and_exchange_names like '%{instrument}%' AND "
                       f"report_date_as_yyyy_mm_dd >= '{start_fmt}' AND "
                       f"report_date_as_yyyy_mm_dd <= '{end_fmt}'"),
            "$order": "report_date_as_yyyy_mm_dd ASC",
            "$limit": 5000
        }

        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        records = response.json()

        if not records:
            print("⚠ No CFTC COT data returned")
            return pd.DataFrame({'Net_Speculative_Position': []})

        raw = pd.DataFrame(records)

        # Extract longs & shorts
        raw['Date'] = pd.to_datetime(raw['report_date_as_yyyy_mm_dd'])
        for col in ['noncomm_positions_long_all', 'noncomm_positions_short_all']:
            if col in raw.columns:
                raw[col] = pd.to_numeric(raw[col], errors='coerce')

        raw['Net_Speculative_Position'] = (
            raw['noncomm_positions_long_all'] - raw['noncomm_positions_short_all']
        )

        df = raw[['Date', 'Net_Speculative_Position']].copy()
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)

        # Deduplicate by taking mean for same dates
        if df.index.duplicated().any():
            df = df.groupby(df.index).mean()

        return df
    except Exception as e:
        print(f"⚠ CFTC COT error: {e}")
        return pd.DataFrame({'Net_Speculative_Position': []})


def fetch_eia_crude_stocks(start, end, config):
    """US Crude Oil Stocks from EIA API v2 (WCESTUS1)."""
    api_key = config.get('EIA_API_KEY')
    if not api_key:
        return pd.DataFrame({'Crude_Stocks_1000bbl': []})
    try:
        url = (f"https://api.eia.gov/v2/petroleum/stoc/wstk/data/"
               f"?api_key={api_key}&frequency=weekly&data[0]=value"
               f"&facets[series][]=WCESTUS1"
               f"&sort[0][column]=period&sort[0][direction]=desc"
               f"&offset=0&length=5000")
        resp = requests.get(url, timeout=30)
        data = resp.json()
        if 'response' in data and 'data' in data['response']:
            df = pd.DataFrame(data['response']['data'])
            df['period'] = pd.to_datetime(df['period'])
            df = df.set_index('period').sort_index()
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            return df[['value']].rename(columns={'value': 'Crude_Stocks_1000bbl'})
    except Exception as e:
        print(f"⚠ EIA Crude Stocks error: {e}")
    return pd.DataFrame({'Crude_Stocks_1000bbl': []})


def fetch_rig_count(start, end, config):
    """US Oil Rig Count from Rigcount_final.csv (Baker Hughes, monthly)."""
    data_dir = config.get('DATA_DIR', 'data')
    rig_path = os.path.join(data_dir, 'Rigcount_final.csv')

    if not os.path.exists(rig_path):
        print(f"⚠ CSV NOT FOUND: {rig_path}")
        return pd.DataFrame({'US_Oil_Rigs': []})

    try:
        rig_raw = pd.read_csv(rig_path)

        # Filter: United States + Oil
        us_mask  = rig_raw['Country'].str.upper().str.strip() == 'UNITED STATES'
        oil_mask = rig_raw['DrillFor'].str.upper().str.strip() == 'OIL'
        rig_us   = rig_raw[us_mask & oil_mask].copy()

        # Build date from Year + Month
        rig_us['Date'] = pd.to_datetime(
            rig_us['Year'].astype(str) + '-' +
            rig_us['Month'].astype(str).str.zfill(2) + '-01'
        )

        # Sum across locations per month
        df = (rig_us.groupby('Date')['Rig Count Value']
              .sum()
              .to_frame('US_Oil_Rigs'))
        df.sort_index(inplace=True)
        df = df.loc[start:end]
        return df
    except Exception as e:
        print(f"⚠ Rig Count error: {e}")
        return pd.DataFrame({'US_Oil_Rigs': []})


# ── BAND 3: SLOW (Monthly → resampled) ─────────────────────────────────────

def fetch_spr(start, end, config):
    """US Strategic Petroleum Reserve from EIA API v2 (WCSSTUS1)."""
    api_key = config.get('EIA_API_KEY')
    if not api_key:
        return pd.DataFrame({'SPR_Stocks_1000bbl': []})
    try:
        url = "https://api.eia.gov/v2/petroleum/stoc/wstk/data/"
        params = {
            "api_key"           : api_key,
            "frequency"         : "weekly",
            "data[0]"           : "value",
            "facets[series][]"  : "WCSSTUS1",
            "start"             : (start.strftime('%Y-%m-%d')
                                   if hasattr(start, 'strftime') else start),
            "end"               : (end.strftime('%Y-%m-%d')
                                   if hasattr(end, 'strftime') else end),
            "sort[0][column]"   : "period",
            "sort[0][direction]": "asc",
            "offset"            : 0,
            "length"            : 5000
        }
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        records = data.get("response", {}).get("data", [])

        if not records:
            return pd.DataFrame({'SPR_Stocks_1000bbl': []})

        df = pd.DataFrame(records)
        df = df[["period", "value"]].rename(
            columns={"period": "Date", "value": "SPR_Stocks_1000bbl"})
        df["Date"] = pd.to_datetime(df["Date"])
        df["SPR_Stocks_1000bbl"] = pd.to_numeric(
            df["SPR_Stocks_1000bbl"], errors="coerce")
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        print(f"⚠ SPR error: {e}")
        return pd.DataFrame({'SPR_Stocks_1000bbl': []})


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE REGISTRY
# ═════════════════════════════════════════════════════════════════════════════
# To add a new feature:
#   1. Write a fetch function above: fetch_xxx(start, end, config) → DataFrame
#   2. Append a dict below. Done!
# ═════════════════════════════════════════════════════════════════════════════

FEATURES = [
    # ── FAST (Daily) ─────────────────────────────────────────────────────
    {
        "name"       : "WTI_Close",
        "band"       : "fast",
        "frequency"  : "daily",
        "ffill_limit": 0,
        "is_base"    : True,
        "fetch"      : fetch_wti,
        "source"     : "Yahoo Finance CL=F",
    },
    {
        "name"       : "Brent_Close",
        "band"       : "fast",
        "frequency"  : "daily",
        "ffill_limit": 0,
        "is_base"    : True,
        "fetch"      : fetch_brent,
        "source"     : "Yahoo Finance BZ=F",
    },
    {
        "name"       : "OVX",
        "band"       : "fast",
        "frequency"  : "daily",
        "ffill_limit": 0,
        "is_base"    : False,
        "fetch"      : fetch_ovx,
        "source"     : "FRED OVXCLS",
    },
    {
        "name"       : "USD_Index",
        "band"       : "fast",
        "frequency"  : "daily",
        "ffill_limit": 0,
        "is_base"    : False,
        "fetch"      : fetch_usd_index,
        "source"     : "Yahoo Finance DX-Y.NYB",
    },

    # ── MEDIUM (Weekly / Monthly) ────────────────────────────────────────
    {
        "name"       : "Crack_3_2_1",
        "band"       : "medium",
        "frequency"  : "daily",      # calculated daily from futures
        "ffill_limit": 7,
        "is_base"    : False,
        "fetch"      : fetch_crack_spread,
        "source"     : "Yahoo Finance RB=F + HO=F",
    },
    {
        "name"       : "Net_Speculative_Position",
        "band"       : "medium",
        "frequency"  : "weekly",
        "ffill_limit": 7,
        "is_base"    : False,
        "fetch"      : fetch_cot_net_spec,
        "source"     : "CFTC Socrata API",
    },
    {
        "name"       : "Crude_Stocks_1000bbl",
        "band"       : "medium",
        "frequency"  : "weekly",
        "ffill_limit": 7,
        "is_base"    : False,
        "fetch"      : fetch_eia_crude_stocks,
        "source"     : "EIA API WCESTUS1",
    },
    {
        "name"       : "US_Oil_Rigs",
        "band"       : "medium",
        "frequency"  : "monthly",
        "ffill_limit": 30,
        "is_base"    : False,
        "fetch"      : fetch_rig_count,
        "source"     : "Rigcount_final.csv",
    },

    # ── SLOW (Monthly) ──────────────────────────────────────────────────
    {
        "name"       : "SPR_Stocks_1000bbl",
        "band"       : "slow",
        "frequency"  : "weekly",
        "ffill_limit": 30,
        "is_base"    : False,
        "fetch"      : fetch_spr,
        "source"     : "EIA API WCSSTUS1",
    },
]
