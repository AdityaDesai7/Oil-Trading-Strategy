# PetroQuant 🛢️

**Quantitative Oil Trading Strategy Platform**

An end-to-end Python system for oil market analysis, featuring a multi-source data pipeline, two ML/RL trading strategies, a walk-forward backtesting engine, and a premium interactive dashboard.

---

## 📁 Project Structure

```
PetroQuant/
│
├── .env                          # API keys (FRED, EIA) — DO NOT COMMIT
│
├── features.py                   # Feature Registry — all 9 oil market data sources
├── oil_data_pipeline.py          # Original monolithic data pipeline
├── oil_data_pipeline_new.py      # Modular pipeline (uses features.py registry)
├── oil_data_pipeline_second.py   # Backup/variant of the modular pipeline
│
├── strategy.py                   # Trading strategies (HMM+XGBoost, TD(0) RL)
├── dashboard.py                  # Backtesting engine + Plotly dashboard renderer
├── run_strategy.py               # Single entry point: Pipeline → Strategy → Backtest → Dashboard
│
├── main.py                       # Health-check & data validation script
├── fun.py                        # Utility: Baker Hughes rig count fetcher
├── rig.py                        # Utility: FRED rig count fetcher
├── daily_tracker.py              # (Placeholder for daily tracking)
│
├── data/                         # Raw/processed data files
│   └── Rigcount_final.csv        # Baker Hughes rig count historical data
│
├── output/                       # Generated dashboards & cached feature CSVs
│   ├── dashboard_hmm_xgb.html    # HMM+XGBoost strategy dashboard
│   ├── dashboard_td0_rl.html     # TD(0) RL strategy dashboard
│   └── master_oil_features_*.csv # Timestamped pipeline output caches
│
└── venv/                         # Python virtual environment
```

---

## 🔧 How It Works

The platform follows a **4-stage pipeline** architecture:

```
 ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
 │  1. DATA      │ ──▶ │  2. STRATEGY  │ ──▶ │  3. BACKTEST  │ ──▶ │  4. DASHBOARD │
 │  PIPELINE     │     │  ENGINE       │     │  ENGINE       │     │  RENDERER     │
 └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

### Stage 1: Data Pipeline (`features.py` + `oil_data_pipeline_new.py`)

Fetches, cleans, and merges **9 oil market features** from multiple APIs into a single `master_df` DataFrame:

| # | Feature | Band | Frequency | Source |
|---|---------|------|-----------|--------|
| 1 | **WTI_Close** | Fast | Daily | Yahoo Finance (CL=F) |
| 2 | **Brent_Close** | Fast | Daily | Yahoo Finance (BZ=F) |
| 3 | **OVX** (Oil Volatility Index) | Fast | Daily | FRED API |
| 4 | **USD_Index** (DXY) | Fast | Daily | Yahoo Finance |
| 5 | **Crack_3_2_1** (3-2-1 Crack Spread) | Medium | Daily | Yahoo Finance (computed) |
| 6 | **Net_Speculative_Position** | Medium | Weekly | CFTC Socrata API |
| 7 | **Crude_Stocks_1000bbl** | Medium | Weekly | EIA API v2 |
| 8 | **US_Oil_Rigs** | Medium | Monthly | Baker Hughes CSV |
| 9 | **SPR_Stocks_1000bbl** | Slow | Weekly | EIA API v2 |

**Key design:**
- Features are organized into **3 signal bands** — Fast (daily), Medium (weekly), Slow (monthly)
- The pipeline uses **24-hour caching** to avoid redundant API calls
- Lower-frequency data is forward-filled to align with the daily index
- Adding a new feature requires only writing a `fetch_xxx()` function and appending to the `FEATURES` list

---

### Stage 2: Trading Strategies (`strategy.py`)

All strategies inherit from a common `BaseStrategy` abstract class. Two strategies are implemented:

#### Strategy A: HMM + XGBoost (Regime-Aware)
- **Regime Detection:** 3-state Hidden Markov Model classifies the market into `BULL`, `PANIC`, or `CHOPPY` regimes using return and volatility features
- **Signal Generation:** Walk-forward XGBoost classifier trained on engineered features (momentum, volatility, spreads, regime) to predict N-day forward returns
- **Forecasting:** Multi-horizon return forecasts (1, 7, 15, 30, 60, 90, 180 days) via conditional historical returns per regime
- **Walk-Forward Training:** Retrains every 63 trading days (~1 quarter) with expanding window

#### Strategy B: TD(0) Reinforcement Learning
- **Algorithm:** Semi-Gradient TD(0) with Linear Function Approximation
- **State Space:** Continuous features (momentum, volatility, regime) normalized via rolling z-scores
- **Action Space:** Long (+1), Flat (0), Short (-1)
- **Reward:** Differential Sharpe Ratio with drawdown penalties to optimize risk-adjusted returns
- **Exploration:** Softmax (Boltzmann) exploration with decaying temperature
- **Walk-Forward Learning:** Learns and adapts online as new data arrives

---

### Stage 3: Backtesting Engine (`dashboard.py` → `Backtester`)

Simulates strategy performance on out-of-sample data:

- Computes daily strategy returns from signal × market return
- Tracks equity curve, drawdown, and cumulative P&L
- Calculates key metrics:
  - **Total Return** & **Annualized Return**
  - **Sharpe Ratio** & **Calmar Ratio**
  - **Max Drawdown**
  - **Win Rate** & **Profit Factor**
  - **Total Trades** & **Trading Days**

---

### Stage 4: Dashboard (`dashboard.py` → `StrategyDashboard`)

Renders a **premium 10-panel interactive Plotly dashboard** saved as standalone HTML:

- Equity curve with regime shading
- Buy/Sell signal overlay on price chart
- Drawdown chart
- Rolling Sharpe ratio
- Feature importance rankings
- Multi-horizon return forecasts
- Signal distribution analysis
- Performance metrics summary

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install pandas numpy yfinance fredapi requests plotly scikit-learn xgboost hmmlearn python-dotenv
```

### 2. Configure API Keys

Create a `.env` file in the project root:

```env
FRED_API_KEY=your_fred_api_key_here
EIA_API_KEY=your_eia_api_key_here
```

- **FRED API Key:** Get one free at [https://fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)
- **EIA API Key:** Get one free at [https://www.eia.gov/opendata/register.php](https://www.eia.gov/opendata/register.php)

### 3. Run the Full Pipeline

```bash
# Run all strategies end-to-end (data → strategy → backtest → dashboard)
python run_strategy.py
```

This will:
1. Build `master_df` from all 9 API sources (or use cached data)
2. Run both HMM+XGBoost and TD(0) RL strategies
3. Backtest each strategy
4. Save interactive HTML dashboards to `output/`

### 4. View Results

Open the generated dashboards in your browser:
- `output/dashboard_hmm_xgb.html` — HMM + XGBoost results
- `output/dashboard_td0_rl.html` — TD(0) RL results

---

## 🔍 Other Scripts

| Script | Purpose |
|--------|---------|
| `main.py` | Run a **data health check** — validates staleness, coverage, and volatility pulse |
| `fun.py` | Standalone Baker Hughes rig count fetcher (Excel download) |
| `rig.py` | Standalone FRED-based rig count fetcher (API) |

---

## 📊 Data Flow Diagram

```
Yahoo Finance ──┐
  (CL=F, BZ=F,  │
   RB=F, HO=F,  │     ┌─────────────────┐     ┌───────────────┐
   DX-Y.NYB)    ├────▶│                 │     │               │
                 │     │  features.py    │     │  strategy.py  │
FRED API ────────┤     │  (9 fetchers)   │────▶│               │
  (OVX)          │     │       ↓         │     │  HMM+XGBoost  │
                 │     │  oil_data_      │     │  TD(0) RL     │
CFTC Socrata ────┤     │  pipeline_new.py│     │       ↓       │     ┌────────────┐
  (COT data)     │     │  (merge & clean)│     │  Signals +    │────▶│ dashboard  │
                 │     │       ↓         │     │  Forecasts    │     │ .py        │
EIA API v2 ──────┤     │  master_df      │     └───────────────┘     │            │
  (Stocks, SPR)  │     │  (daily index)  │                           │ Backtester │
                 │     └─────────────────┘                           │ Dashboard  │
Baker Hughes ────┘                                                   │     ↓      │
  (Rig Count CSV)                                                    │ HTML files │
                                                                     └────────────┘
```

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **pandas / NumPy** — Data manipulation
- **yfinance** — Yahoo Finance market data
- **fredapi** — FRED economic data
- **requests** — API calls (EIA, CFTC)
- **scikit-learn** — Preprocessing & metrics
- **XGBoost** — Gradient boosting classifier
- **hmmlearn** — Hidden Markov Model
- **Plotly** — Interactive dashboards
- **python-dotenv** — Environment variable management

---

## 📝 License

This project is for educational and research purposes only. Not financial advice.
