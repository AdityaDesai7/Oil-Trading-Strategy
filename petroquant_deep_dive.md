# PetroQuant — Complete Deep Dive

> **Everything about this Oil Trading Strategy, explained for beginners with examples.**
> Updated: March 2026

---

## Table of Contents

1. [What Does This System Do?](#1-big-picture)
2. [File Structure — What Each File Does](#2-files)
3. [Part A — The Data Pipeline (Getting the Data)](#3-pipeline)
4. [Part B — The Trading Strategy (The Brain)](#4-strategy)
5. [Part C — The Volatility Engine (GARCH + ATR + OVX)](#5-volatility-engine)
6. [Part D — Backtesting & Dashboard (Did It Work?)](#6-backtesting)
7. [Part E — Terminal Output Explained Line-by-Line](#7-terminal-walkthrough)
8. [Glossary — Every Term Explained](#8-glossary)

---

## 1. What Does This System Do? {#1-big-picture}

Imagine you trade crude oil. Every day you need to answer:

> **"Should I BUY, SELL, or HOLD oil today?"**

PetroQuant answers this by:

1. **Fetching data** — Oil prices, volatility, inventories, dollar strength, etc.
2. **Detecting the market mood** — Is the market calm (BULL), uncertain (CHOPPY), or panicking (PANIC)?
3. **Predicting direction** — Using AI to predict if oil goes UP or DOWN in the next 5 days
4. **Generating signals** — BUY / SELL / HOLD with confidence levels
5. **Forecasting prices** — Expected WTI price for 1 day to 6 months, with 95% confidence ranges
6. **Testing on history** — "If I had followed these signals in the past, how much money would I have made?"

Here's the flow:

```
DATA PIPELINE         STRATEGY              BACKTEST            DASHBOARD
(Fetch & Clean)  -->  (HMM + XGBoost)  -->  (Test on Past)  --> (Visual Report)
                                 |
                      VOLATILITY ENGINE
                      (GARCH + ATR + OVX)
                      for Price Ranges
```

---

## 2. File Structure {#2-files}

| File | What It Does | Think of It As... |
|------|-------------|-------------------|
| `features.py` | Defines what data to fetch and how | The *menu card* — lists every ingredient |
| `oil_data_pipeline_new.py` | Fetches, cleans, and merges all data | The *kitchen* — prepares the ingredients |
| `strategy.py` | The AI brain — HMM + XGBoost + VolatilityEngine | The *chef* — makes the trading decisions |
| `dashboard.py` | Backtesting engine + visual dashboard | The *food critic* — tests and rates the chef |
| `run_strategy.py` | Connects everything together | The *restaurant manager* — coordinates everyone |

**Other files:**
- `.env` — Stores your API keys (FRED_API_KEY, EIA_API_KEY)
- `data/Rigcount_final.csv` — Baker Hughes rig count data (read from local CSV)
- `output/` — Where the dashboard HTML and cached CSVs are saved

---

## 3. Part A — The Data Pipeline {#3-pipeline}

### What is a Data Pipeline?

A data pipeline is like a kitchen conveyor belt:

1. **Raw ingredients arrive** — Data from Yahoo Finance, FRED, EIA, CFTC, CSV files
2. **They get washed** — Dates standardised, gaps filled, duplicates removed
3. **Combined onto one plate** — All features joined into one big table called `master_df`

The result is a table where:
- Each **row** = one trading day (e.g., Jan 15, 2024)
- Each **column** = one piece of data (e.g., WTI price, OVX fear index)

**Example of what master_df looks like:**

| Date | WTI_Close | Brent_Close | OVX | USD_Index | Crude_Stocks | ... |
|------|-----------|-------------|------|-----------|-------------|-----|
| 2024-01-02 | 70.38 | 76.04 | 28.4 | 101.3 | 431,000 | ... |
| 2024-01-03 | 69.15 | 74.88 | 29.1 | 102.0 | 431,000 | ... |
| 2024-01-04 | 72.70 | 78.25 | 27.2 | 101.8 | 427,000 | ... |

Notice how `Crude_Stocks` shows the same value for Jan 2 and 3? That's because inventory data comes weekly — the pipeline "forward-fills" it (uses the last known value until new data arrives).

---

### The 9 Features — What Data We Fetch and Why

#### 1. WTI_Close — West Texas Intermediate Crude Price
- **Source**: Yahoo Finance (ticker: `CL=F`)
- **Updates**: Every trading day
- **Why**: This IS the oil price. It's what we're trying to predict and trade.
- **Example**: WTI was $47 in Jan 2021, hit $120+ in June 2022, dropped to $67 by late 2025.

#### 2. Brent_Close — International Oil Benchmark
- **Source**: Yahoo Finance (ticker: `BZ=F`)
- **Updates**: Every trading day
- **Why**: Brent is the international oil price. The *spread* (WTI minus Brent) tells us about supply/demand balance.
- **Example**: If WTI = $70 and Brent = $75, the $5 gap hints at US supply problems.

#### 3. OVX — Oil Volatility Index (Fear Gauge)
- **Source**: FRED API (series: `OVXCLS`)
- **Updates**: Every trading day
- **Why**: OVX measures how scared traders are. High OVX = "I expect wild swings." Low OVX = "Everything is calm."
- **Example**: OVX was 300+ during COVID crash (March 2020). Normal = 25-40.

> **Think of OVX like a weather forecast:** Low OVX = "clear skies ahead." High OVX = "storm warning — buckle up."

#### 4. USD_Index (DXY) — Dollar Strength
- **Source**: Yahoo Finance (ticker: `DX-Y.NYB`)
- **Updates**: Every trading day
- **Why**: Oil is priced in dollars. When the dollar gets STRONGER, oil usually gets CHEAPER. This is because foreign buyers need more of their currency to afford the same dollar-priced barrel.
- **Example**: Dollar goes from 100 to 105 (+5%). Oil often drops because it's more expensive for Europe and Asia.

#### 5. Crack_3_2_1 — Refinery Profit Margin
- **Source**: Calculated from Yahoo Finance (gasoline `RB=F` + heating oil `HO=F` + crude `CL=F`)
- **Updates**: Every trading day
- **Why**: Measures how profitable it is for refineries to turn crude oil into gasoline and diesel.

**The formula (simplified):**
```
Crack Spread = (2 x Gasoline Price x 42 + 1 x Heating Oil Price x 42 - 3 x Crude Price) / 3
                    ↑ the "x 42" converts gallons to barrels (1 barrel = 42 gallons)
```

**Why it matters:** High crack spread = refiners make lots of money = they buy MORE crude = bullish for oil.

#### 6. Net_Speculative_Position — What Are Big Traders Betting On?
- **Source**: CFTC (Commodity Futures Trading Commission) via Socrata API
- **Updates**: Weekly (every Friday)
- **Why**: Shows what hedge funds and big speculators are doing. Are they betting on oil going UP or DOWN?

```
Net Position = Long Contracts - Short Contracts
  +150,000 = heavily bullish (more longs than shorts)
  -120,000 = heavily bearish (more shorts than longs)
```

> **Think of it like a poll:** If 80% of big traders are betting UP, that's useful information. But extreme consensus can also signal a reversal ("everyone who wants to buy has already bought").

#### 7. Crude_Stocks_1000bbl — US Oil Inventories
- **Source**: EIA API (series: `WCESTUS1`)
- **Updates**: Weekly
- **Why**: How much oil is sitting in storage tanks? Rising = too much supply = bearish. Falling = demand exceeds supply = bullish.
- **Example**: Stocks drop from 450M to 440M barrels = 10M barrels of demand not met by production = bullish.

#### 8. US_Oil_Rigs — Baker Hughes Rig Count
- **Source**: Local CSV file (`Rigcount_final.csv`)
- **Updates**: Monthly
- **Why**: More rigs = more future production = potentially bearish long-term. Fewer rigs = less future supply = bullish.

#### 9. SPR_Stocks_1000bbl — Strategic Petroleum Reserve
- **Source**: EIA API (series: `WCSSTUS1`)
- **Updates**: Weekly
- **Why**: The SPR is the US government's emergency oil reserve. When the government RELEASES oil from SPR = more supply = bearish. When they REFILL = more demand = bullish.
- **Example**: In 2022, Biden released ~180M barrels from SPR to fight high gas prices.

---

### How build_master_df() Works

This is the main function in `oil_data_pipeline_new.py`. When you run it:

**Step 1: Check Cache**
```
Is there a recent CSV (< 24 hours old) in /output/?
  YES --> Load it instead of re-fetching (saves time!)
  NO  --> Continue to Step 2
```

**Step 2: Fetch All 9 Features**
```
For each feature:
  Call its fetch() function --> get data from Yahoo/FRED/EIA/CSV
  If it fails --> skip it, continue with the rest
```

**Step 3: Build the Base**
```
Combine WTI + Brent (the "base" features)
Only keep days where BOTH have data
```

**Step 4: Join Everything Else**
```
For each remaining feature:
  Clean the data (remove duplicates, fix dates)
  Join it onto the base using dates
  Forward-fill gaps (e.g., weekly data fills Mon-Thu with Friday's value)
```

**Step 5: Save**
```
Save as CSV: output/master_oil_features_20260309_152320.csv
Print a report showing how complete each column is
```

### Forward-Fill Explained

Many features don't update daily. The pipeline fills gaps:

```
Raw weekly COT data:           After forward-fill:
Fri Jan 3  | 150,000           Fri Jan 3  | 150,000
Sat Jan 4  | (empty)           Sat Jan 4  | 150,000  <-- filled
Mon Jan 6  | (empty)           Mon Jan 6  | 150,000  <-- filled
Tue Jan 7  | (empty)           Tue Jan 7  | 150,000  <-- filled
Fri Jan 10 | 155,000           Fri Jan 10 | 155,000  <-- new data!
```

---

## 4. Part B — The Trading Strategy {#4-strategy}

The strategy lives in `strategy.py`. It uses two AI models working together:

```
HMM (Hidden Markov Model)          XGBoost (Gradient Boosting)
"What MOOD is the market in?"  -->  "Given this mood + all data,
                                     will price go UP or DOWN?"
 BULL   = calm, trending up
 CHOPPY = uncertain, sideways        Output: 73% chance UP
 PANIC  = volatile, scary            --> SIGNAL = BUY
```

### Step 1: Feature Engineering — Making Raw Data Useful

Raw numbers aren't enough for AI. We need to transform them into *patterns*. Here's what the strategy creates:

#### Log Returns — "How much did the price change?"

```
Formula: LogReturn = ln(Price_today / Price_yesterday)

Example:
  Yesterday: WTI = $70.00
  Today:     WTI = $71.40
  LogReturn = ln(71.40 / 70.00) = +1.98%

  "Oil went up about 2% today"
```

We compute this for WTI (1-day and 5-day) and Brent (1-day and 5-day).

Why log instead of regular percentage? Log returns are *additive* — you can sum them across days. A +10% followed by -10% using log returns nets to zero. With regular percentages, $100 -> $110 -> $99 = you actually lost $1!

#### Z-Scores — "Is this value unusual?"

```
Z-Score = (Today's value - Average over last 20 days) / Standard Deviation

Z-Score = 0   --> perfectly average
Z-Score = +2  --> unusually HIGH (2 standard deviations above)
Z-Score = -2  --> unusually LOW
```

We compute Z-scores for the WTI-Brent spread, speculative positions, and crack spread.

**Example:** If the WTI-Brent spread is normally -$3.50 but today it's -$5.10:
```
Z-Score = (-5.10 - (-3.50)) / 0.80 = -2.0
"The spread is 2 standard deviations below normal — something unusual is happening"
```

#### Momentum — "Is there a trend?"

```
WTI_Mom_10 = Price change over last 10 days (%)
WTI_Mom_20 = Price change over last 20 days (%)

Example: Oil was $68 ten days ago, now $72
  Mom_10 = (72-68)/68 = +5.9% --> Strong upward trend
```

#### Realised Volatility — "How jumpy has the price been?"

```
Formula: RealizedVol_20d = Std Dev of daily returns over 20 days x sqrt(252)

The x sqrt(252) converts daily vol to annual (252 trading days in a year).

Example:
  Daily std dev = 1.5%
  Annual vol = 1.5% x 15.87 = 23.8%
  "Oil is swinging about 24% per year — that's fairly volatile"
```

#### All Engineered Features Summary

| Feature | What It Captures |
|---------|-----------------|
| WTI_LogRet_1d, 5d | How much oil moved recently |
| Brent_LogRet_1d, 5d | How Brent oil moved |
| Spread_Zscore | Is WTI-Brent gap unusual? |
| WTI_Mom_10, 20 | Price trend direction |
| RealizedVol_20d | How jumpy prices are |
| OVX_Chg_1d, 5d | Is fear rising or falling? |
| USD_Chg_1d, 5d, ROC_10 | Dollar strength changes |
| NSP_Chg_5d, NSP_Zscore | Speculator positioning |
| Crude_Chg_5d | Inventory building/draining? |
| Rigs_Chg_5d | Production capacity changing? |
| SPR_Chg_5d | Government adding/releasing? |
| Crack_Zscore, Crack_Chg_5d | Refinery margins |

After computing all features:
1. Replace infinity values with NaN
2. Drop all rows with any missing data
3. Drop columns with zero variance (no information)

Result: **~20 clean features** ready for machine learning.

---

### Step 2: HMM Regime Detection — "What Mood Is the Market In?"

#### What is a Hidden Markov Model?

Imagine the market is a person with three moods, but you can't directly see their mood. You CAN observe:
- How much the price moved today (returns)
- How jumpy prices have been (volatility)

From these observations, HMM *guesses* the hidden mood:

```
BULL   = Calm mood    | Small daily moves | Low volatility
CHOPPY = Mixed mood   | Medium moves      | Moderate volatility
PANIC  = Scared mood  | Big swings        | High volatility
```

#### How It Works (Analogy)

You're outside a room. Someone inside can be happy, neutral, or angry. You can't see them but you hear their voice (loud/quiet, positive/negative). Based on that, you guess their mood. That's HMM!

#### What the Code Actually Does

```python
# Feed HMM two signals for each day
X_hmm = [daily_returns, realised_volatility]

# HMM learns:
# 1. What returns/vol look like in each state
# 2. How likely each state transitions to another
# 3. Which state each day belongs to

hmm_model.fit(data)
hidden_states = hmm_model.predict(data)  # [0, 0, 1, 2, 2, 0, ...]
```

#### Regime Labelling

HMM outputs numbers (0, 1, 2). The code maps them to names:
1. Sort states by volatility (highest first)
2. Highest vol = **PANIC**
3. Of the remaining two, higher return = **BULL**, other = **CHOPPY**

**Example output:**
```
[+] BULL     +0.00080 (+20.2%/yr)    0.1800    950 days (51.9%)
[~] CHOPPY   +0.00010 (+2.5%/yr)     0.2500    468 days (25.6%)
[!] PANIC    -0.00100 (-25.2%/yr)    0.4200    412 days (22.5%)
```

Translation: "The market spent 52% of the time calm and trending up, 26% uncertain, and 22% in panic mode."

---

### Step 3: Walk-Forward XGBoost — "Will Price Go Up or Down?"

#### What is XGBoost?

XGBoost is one of the most powerful AI algorithms for tabular data. Think of it like building a team of 300 simple decision trees:

```
Tree 1: "If OVX > 35 AND momentum is negative --> predict DOWN"
Tree 2: focuses on what Tree 1 got WRONG
Tree 3: focuses on what Tree 2 got WRONG
...
Tree 300: final corrections

All 300 trees VOTE together for the final answer.
```

#### What Is It Predicting?

```
Target = "Did oil go UP in the next 5 trading days?"

Example:
  Today (Jan 10): WTI = $72.00
  5 days later (Jan 17): WTI = $74.50
  Return = +3.47% --> Target = 1 (UP)
  If it had dropped to $70.00 --> Target = 0 (DOWN)
```

#### Walk-Forward: Why It Matters

**The Problem:** If you train on ALL your data and then test on part of it, you're cheating — the model has "seen" the test answers.

**Walk-Forward solves this:**

```
Time ---->

Fold 1: [===TRAIN (500 days)=====][TEST 63 days]
Fold 2: [========TRAIN (563 days)========][TEST 63 days]
Fold 3: [===========TRAIN (626 days)===========][TEST 63 days]
...

Rules:
- NEVER train on future data
- Always test on data the model has NEVER seen
- Retrain every 63 days (~1 quarter) to adapt
```

This simulates real trading: you only know the past, and you're predicting the unknown future.

#### From Probability to Signal

XGBoost outputs a probability (0 to 100%):

```
Probability > 55%  -->  BUY   "Model is fairly confident price goes UP"
Probability < 45%  -->  SELL  "Model is fairly confident price goes DOWN"
45% to 55%         -->  HOLD  "Model is uncertain — stay out"
```

**Why not use 50% as the cutoff?** Because 51% is basically a coin flip — not worth trading on. The 45-55% "no-trade zone" ensures we only act when the AI has real conviction.

---

### Regime-Aware Position Sizing

Not all signals are created equal. A BUY signal in a calm BULL market is more reliable than a BUY signal during PANIC.

```
Position_Size = Signal x Regime_Mult x Confidence_Factor

Regime multipliers:
  BULL   = 1.00x  (full size — market is calm, go for it)
  CHOPPY = 0.50x  (half size — uncertain, be cautious)
  PANIC  = 0.25x  (quarter size — wild swings, protect capital)

Confidence_Factor:
  Based on how far the probability is from 50%
  73% probability --> high confidence --> factor = 0.96
  52% probability --> low confidence  --> factor = 0.54
  Scales from 0.5 (minimum) to 1.0 (maximum)
```

**Example:**
```
Signal = BUY (+1), Regime = CHOPPY, Probability = 68%

Regime_Mult = 0.50
Confidence_Factor = 0.5 + 0.5 x (|0.68 - 0.50| x 2) = 0.5 + 0.5 x 0.36 = 0.68

Position_Size = 1 x 0.50 x 0.68 = 0.34

"Buy, but only with 34% of your capital (half size due to choppy market,
 further reduced because confidence isn't maximum)"
```

---

## 5. Part C — The Volatility Engine (GARCH + ATR + OVX) {#5-volatility-engine}

### The Problem

When the strategy says "WTI will be $60 in 30 days," we need to know: **how confident is this?** Could it actually be $55 or $65? Or could it be $40 or $80?

The answer depends on **volatility** — how much the price swings. And different time horizons have DIFFERENT volatility:
- 1 day: price barely moves (~1%)
- 30 days: larger moves possible (~5-10%)
- 180 days: anything can happen (~15-30%)

### The Old Way (Simple)

```
Price Range = Current Price x (1 +/- 1.96 x daily_vol x sqrt(horizon))
```

This is like using one weather station for every city. It works, but it's crude.

### The New Way: 3-Layer Composite Volatility

PetroQuant now uses THREE different volatility measures and blends them:

#### Layer 1: GARCH(1,1) — Conditional Volatility

**What is GARCH?** It stands for *Generalized Autoregressive Conditional Heteroskedasticity*. Don't worry about the name — here's what it does:

**Key insight:** Volatility clusters. After a big move (up OR down), more big moves tend to follow. After small moves, more small moves follow. GARCH captures this.

```
GARCH says: "Today's volatility depends on:
  - Yesterday's volatility (does the storm continue?)
  - Yesterday's shock (did something big happen?)
  - The long-run average volatility (storms always pass eventually)"

Parameters:
  alpha = 0.05  (5% weight on yesterday's shock)
  beta  = 0.90  (90% weight on yesterday's volatility)
  persistence = alpha + beta = 0.95

Translation: "95% of today's storm carries into tomorrow.
              The remaining 5% mean-reverts toward normal."
```

**Why it's better for medium horizons (15-30 days):** GARCH knows that high volatility decays back to normal over time. A flash crash doesn't mean 30-day volatility should explode — the storm will pass.

#### Layer 2: ATR(14) — Average True Range

**What is ATR?** It measures how much the price actually *moves* on a typical day, in dollar terms.

```
ATR = Smoothed average of |daily price change| over 14 days

Example:
  If oil moves about $1.20/day on average, and the price is $60:
  ATR = $1.20
  ATR as fraction = 1.20 / 60 = 2.0% of price

For horizons:
  1-day range = ATR x sqrt(1) = $1.20
  7-day range = ATR x sqrt(7) x decay = ~$2.85
```

**Why it's better for short horizons (1-7 days):** ATR is very responsive to recent price action. If oil has been quiet for 2 weeks (small daily moves), ATR gives a tight range. If it's been wild, ATR widens.

#### Layer 3: OVX Implied Volatility

**What is OVX?** It's the market's OWN estimate of future oil volatility. Just like VIX is the "fear gauge" for stocks, OVX is the fear gauge for oil.

```
OVX = 30 means the market expects oil to swing ~30% over the next year

For a specific horizon:
  vol_h = (OVX / 100) x sqrt(h / 252)

Example (OVX = 30):
  30-day vol = 0.30 x sqrt(30/252) = 0.30 x 0.345 = 10.3%
  "The market expects ~10% swings over the next month"
```

**Why it's better for long horizons (60-180 days):** OVX incorporates ALL information the market knows — geopolitics, OPEC decisions, demand forecasts — things that our models can't easily see.

### How the Three Layers are Blended

Different layers are weighted differently depending on how far ahead you're looking:

| Horizon | GARCH | ATR | OVX(IV) | Logic |
|---------|-------|-----|---------|-------|
| **1 day** | 25% | **50%** | 25% | Short-term: ATR dominates (recent price action) |
| **7 days** | 30% | **40%** | 30% | Still short: ATR still important |
| **15 days** | **40%** | 25% | 35% | Medium: GARCH's clustering matters |
| **30 days** | **40%** | 20% | 40% | Medium: GARCH + IV roughly equal |
| **60 days** | 30% | 15% | **55%** | Longer: market's IV starts to dominate |
| **90 days** | 25% | 10% | **65%** | Long: IV heavily weighted |
| **180 days** | 20% | 10% | **70%** | Very long: what does the OPTIONS market think? |

### Computing the 95% Confidence Interval

```
composite_vol = w_garch x vol_garch + w_atr x vol_atr + w_iv x vol_iv

CI_lower = Current_Price x (1 + Expected_Return - 1.96 x composite_vol)
CI_upper = Current_Price x (1 + Expected_Return + 1.96 x composite_vol)
```

**Why 1.96?** In statistics, 1.96 standard deviations covers 95% of a normal distribution. So there's a 95% chance the actual price falls within this range.

**Example (30-day forecast):**
```
Current price = $58.00, Expected return = +1.0%

GARCH vol = 7.6%, ATR vol = 3.4%, IV vol = 8.8%
Weights: 40% / 20% / 40%

Composite = 0.40 x 7.6% + 0.20 x 3.4% + 0.40 x 8.8%
          = 3.04% + 0.68% + 3.52%
          = 7.24%

Expected price = $58.00 x 1.01 = $58.58
CI_lower = $58.00 x (1 + 0.01 - 1.96 x 0.0724) = $58.00 x 0.868 = $50.34
CI_upper = $58.00 x (1 + 0.01 + 1.96 x 0.0724) = $58.00 x 1.152 = $66.82

"We expect $58.58, but 95% confident price will be between $50.34 and $66.82"
```

Notice: 30-day range is about $16 wide. For 1-day it would be only ~$2. For 180-day it might be $30+. This makes sense — the further you look, the less certain you can be.

---

## 6. Part D — Backtesting & Dashboard {#6-backtesting}

### What is Backtesting?

Backtesting answers: **"If I had followed these signals in the PAST, how much money would I have made?"**

It simulates trading:
1. Start with $1.00
2. Each day, check the signal (BUY/SELL/HOLD)
3. Compute your return based on whether you were right

```
Day    WTI Move    Signal    Your Return    Buy&Hold Return
Mon    +1.5%       BUY(+1)    +1.5%           +1.5%
Tue    -2.0%       SELL(-1)   +2.0%  <--nice! -2.0%
Wed    +0.3%       HOLD(0)     0.0%           +0.3%
Thu    -1.2%       SELL(-1)   +1.2%           -1.2%
Fri    +0.8%       BUY(+1)    +0.8%           +0.8%
                              -------         -------
Week:                         +5.5%           -0.6%

"Strategy made +5.5% while just holding oil lost -0.6%!"
```

### Performance Metrics — What Each One Means

#### The Basics

| Metric | What It Means | Your Result | Good? |
|--------|--------------|-------------|-------|
| **Total Strategy Return** | Total profit following all signals | +148.39% | Excellent |
| **Buy & Hold Return** | What you'd earn just holding oil | -24.44% | (benchmark) |
| **Alpha** | Strategy return MINUS buy-and-hold | +172.84% | Outstanding |
| **Annualised Return** | Average yearly return | +36.78% | Excellent |
| **Annualised Volatility** | How much returns swing per year | 30.05% | Moderate |
| **Trading Days** | Days in the backtest | 732 | ~2.9 years |

#### Risk-Adjusted Ratios

| Metric | Formula (Simplified) | Your Result | What It Means |
|--------|---------------------|-------------|---------------|
| **Sharpe Ratio** | Return / Total Volatility | 1.193 | Good — you get more than 1 unit of return per unit of risk |
| **Sortino Ratio** | Return / *Downside* Volatility only | 1.687 | Like Sharpe but only counts the BAD swings. Higher = better upside |
| **Information Ratio** | Excess Return / Tracking Error | 1.032 | Measures skill vs benchmark. >0.5 is good, >1.0 is excellent |
| **Treynor Ratio** | Return / Beta | 1.956 | Return per unit of MARKET risk. High = low correlation to market |
| **Omega Ratio** | Total Gains / Total Losses | 1.236 | Profit-to-loss ratio. >1.0 = profitable, >1.5 = strong |

#### Pain & Recovery

| Metric | What It Means | Your Result | Interpretation |
|--------|--------------|-------------|----------------|
| **Max Drawdown** | Worst peak-to-trough drop | -27.95% | "At the worst point, you lost 28% from your high" |
| **Calmar Ratio** | Annual Return / Max Drawdown | 1.316 | >1.0 means returns justify the pain |
| **Sterling Ratio** | Annual Return / Avg of Top 3 Drawdowns | 1.856 | More stable than Calmar (uses 3 worst drops, not just 1) |
| **Ulcer Index** | Depth + Duration of drawdowns combined | 9.93 | Lower = less stressful. <10 is good. |

**What is a drawdown?** Imagine your account goes: $100 -> $120 -> $95 -> $80 -> $110.
The drawdown from peak ($120) to trough ($80) = -33%. That's how bad it got.

#### Trade Efficiency

| Metric | What It Means | Your Result |
|--------|--------------|-------------|
| **Win Rate** | % of trades that made money | 55.7% (>50% needed) |
| **Profit Factor** | Gross profit / gross loss | 1.236 |
| **Expectancy** | Average profit per trade | 16.77 bps ($1.68 per $10,000) |
| **Avg Win** | Average gain on winning trades | 157.79 bps |
| **Avg Loss** | Average loss on losing trades | 160.65 bps |
| **Tail Ratio** | Big wins / big losses | 1.081 (your best days > worst days) |

> **What are basis points (bps)?** 1 basis point = 0.01%. So 100 bps = 1%. If you trade $10,000 and earn 16.77 bps, that's $1.68 profit.

#### Market Skill

| Metric | What It Means | Your Result |
|--------|--------------|-------------|
| **Up-Market Capture** | How much of bull-day gains you captured | 35.2% |
| **Down-Market Capture** | How much of bear-day losses you absorbed | 16.3% |
| **Capture Ratio** | Up Capture / Down Capture | 2.163 |

**Translation:** "On days oil went UP, you only captured 35% of the gain (low). BUT on days oil went DOWN, you only lost 16% of the drop (excellent). Your capture ratio of 2.16 means your dodging skill is twice as good as your capturing skill — that's actually great for risk management."

### The Dashboard — 10 Visual Panels

The dashboard is saved as `output/dashboard_hmm_xgb.html` — open it in any browser.

| Panel | What It Shows |
|-------|-------------|
| **(1) Price + Regime Shading** | Full WTI price history with green/yellow/red background showing market mood |
| **(2) Buy/Sell Signal Overlay** | Price chart with green triangles (BUY) and red triangles (SELL) |
| **(3) Equity Curve** | Your strategy growth vs just holding oil. Strategy line should be ABOVE buy-and-hold |
| **(4) Drawdown** | How deep your losses got at each point (red "underwater" chart) |
| **(5) Feature Importance** | Which data features XGBoost relies on most (horizontal bar chart) |
| **(6) Accuracy per Regime** | How accurate the model is in BULL vs CHOPPY vs PANIC markets |
| **(7) Rolling Sharpe** | Sharpe ratio over time — is the strategy improving or deteriorating? |
| **(8) Monthly Returns Heatmap** | Color-coded grid of returns by month and year |
| **(9) Price Forecast + 95% CI** | Expected future prices with error bars showing the volatility-based range |
| **(10) Performance Summary** | Table of all metrics with "What It Means" column |

---

## 7. Part E — Terminal Output Explained Line-by-Line {#7-terminal-walkthrough}

When you run `python run_strategy.py`, here's what every line means:

### Phase 1: Data Pipeline

```
======================================================================
  PETROQUANT -- STRATEGY RUNNER
======================================================================
  Loading data from pipeline...
  (Fetching oil prices, volatility, inventories, and macro data)
```
The runner starts and calls `build_master_df()`.

```
  OIL DATA PIPELINE — Fetching 9 Features
  Range: 2020-11-30 -> 2025-11-29
```
Fetching 5 years of data (you set END_DATE = Nov 29, 2025).

```
-- FAST (Daily) ---
  [1/9] WTI_Close       <- Yahoo Finance CL=F ... OK 1258 rows
  [2/9] Brent_Close      <- Yahoo Finance BZ=F ... OK 1259 rows
  [3/9] OVX              <- FRED OVXCLS ... OK 1258 rows
  [4/9] USD_Index        <- Yahoo Finance DX-Y.NYB ... OK 1258 rows
```
**FAST band** = data that updates every trading day. ~1258 rows = about 5 years of trading days (252/year x 5 = 1260).

```
-- MEDIUM (Weekly/Monthly) ---
  [5/9] Crack_3_2_1       <- Yahoo Finance RB=F + HO=F ... OK 1258 rows
  [6/9] Net_Speculative_Position <- CFTC Socrata API ... OK 261 rows
  [7/9] Crude_Stocks_1000bbl     <- EIA API WCESTUS1 ... OK 2266 rows
  [8/9] US_Oil_Rigs              <- Rigcount_final.csv ... OK 60 rows
```
**MEDIUM band** = weekly/monthly data.
- COT (speculative positions) = 261 rows. That's 261 weeks = ~5 years of weekly reports.
- Rig count = 60 rows. That's 60 months = 5 years of monthly data.
- These get forward-filled to match daily frequency.

```
  [9/9] SPR_Stocks_1000bbl  <- EIA API WCSSTUS1 ... OK
```
Strategic reserve inventory data.

```
  [OK] Loaded 1258 rows x 12 cols
  [OK] Range: 2020-12-01 -> 2025-11-28
```
The final `master_df` has 1258 trading days and 12 columns (9 features + a few derived ones).

### Phase 2: Strategy Step 1 — Feature Engineering

```
  [1/4] Engineering features...
         (Transforming raw market data into patterns the AI can learn from)
    [OK] 20 engineered features from 10 raw variables
         (e.g., 'WTI_LogRet_1d' = how much oil price changed today in %)
    [OK] Clean DataFrame: 1232 rows x 32 cols
         (Each row = one trading day, each column = one data signal)
```
Started with 12 raw columns, created 20 new features (log returns, z-scores, momentum, etc.). Lost 26 rows from the original 1258 due to dropping NaN (the rolling calculations need a "warm-up" period of 20 days).

### Phase 3: Strategy Step 2 — HMM Regime Detection

```
  [2/4] Fitting HMM regime detection...
         (Detecting the market's 'mood' -- is it calm, choppy, or panicking?)
    [OK] HMM converged -- Log-likelihood: -1234.56
```
The HMM successfully fit. "Converged" = the algorithm found a stable solution. The log-likelihood is a technical measure of fit quality (more negative = less perfect, but irrelevant in practice).

```
      Regime     Avg Daily Return   Avg Volatility         Days
      ----------------------------------------------------------------
      [+] BULL     +0.00040 (+10.1%/yr)    0.1850    650 days (52.8%)
      [~] CHOPPY   +0.00005 (+1.3%/yr)     0.2600    340 days (27.6%)
      [!] PANIC    -0.00080 (-20.2%/yr)    0.4100    242 days (19.6%)
```

**How to read this table:**
- **BULL (52.8% of days):** Average daily return = +0.04% (about +10% per year). Low volatility (0.185). This is the calm, uptrending market.
- **CHOPPY (27.6%):** Barely any return (+1.3%/yr). Medium volatility. Market is deciding which way to go.
- **PANIC (19.6%):** LOSING money (-20%/yr). Very high volatility (0.41). This is COVID crashes, oil wars, etc.

### Phase 4: Strategy Step 3 — Walk-Forward XGBoost

```
  [3/4] Walk-forward XGBoost (20 features, 5-day target)...
         (Training an AI on past data, then testing on unseen future data)
         (Retrains every 63 days to adapt to changing markets)
    [OK] 12 folds | Avg OOS accuracy: 52.4%
         (FAIR: >50% means the model is better than a coin flip)
```
The model was retrained 12 times (every quarter). Average out-of-sample accuracy = 52.4%. This might seem low, but:

> **52.4% in financial markets is actually useful.** Think of it like a casino — the house only has a 51-52% edge, but over thousands of hands, they make billions. Same principle here. You don't need to be right every time, you need to be right *slightly more than wrong* consistently.

```
    [OK] Signals: BUY=285 (38.9%) | SELL=227 (31.0%) | HOLD=220 (30.0%)
```
Out of 732 out-of-sample days:
- 285 days: model says BUY (confident price goes up)
- 227 days: model says SELL (confident price goes down)
- 220 days: model says HOLD (not sure enough to trade)

```
    [OK] Regime-Aware Position Sizing active
      Avg position size: 0.42 (1.0 = full, 0.25 = quarter)
```
On average, the strategy uses 42% of full position size. This means it's being cautious — sizing down during CHOPPY and PANIC markets.

### Phase 5: Strategy Step 4 — Multi-Horizon Forecasting

```
  [4/4] Multi-horizon return forecasting ([1, 7, 15, 30, 60, 90, 180])...
         (Uses GARCH + ATR + OVX composite volatility for 95% CI)

    Current WTI Price: $57.95
```

This is the predict-the-future step. The model forecasts where oil will be from 1 day to 180 days.

```
    -- Volatility Engine Layers --
    [OK] GARCH(1,1) fitted
      alpha=0.0500  beta=0.9000  persistence=0.9500
      1-day cond. vol = 0.0180 (1.80%)
    [OK] ATR(14) = $1.04 (1.79% of price)
    [OK] OVX (Implied Vol) = 32.5% annualized
```

The three volatility layers are initialised:
- **GARCH:** persistence = 0.95 means volatility is VERY sticky (95% carries to tomorrow). 1-day vol = 1.80%.
- **ATR:** Average daily move = $1.04 (1.79% of $57.95). This is the "noise floor."
- **OVX:** Market expects 32.5% annual volatility. This is what options traders are pricing in.

```
    Horizon    Price Range (95% CI)     Expected    Return    Vol(G/A/I->C)    Confidence
    -----------------------------------------------------------------------------------
      1d       $56.85 -- $59.07        $57.97      +0.03%   1.8/1.8/2.0->1.9% 51.2% (LOW)    - Neutral
      7d       $53.80 -- $61.60        $57.61      -0.58%   4.7/4.1/5.3->4.7% 45.8% (LOW)    - Neutral
     15d       $51.13 -- $67.10        $59.12      +0.97%   7.6/3.4/8.8->7.0% 71.2% (HIGH)   ^ Bullish
     30d       $49.50 -- $68.10        $59.68      +1.97%   10.2/2.3/12.6->9.5% 65.7% (HIGH) ^ Bullish
```

**How to read each line (using the 30d example):**
- **30d** = forecast for 30 days from now
- **$49.50 -- $68.10** = 95% CI. "We're 95% confident oil will be between $49.50 and $68.10 in 30 days."
- **$59.68** = expected (most likely) price
- **+1.97%** = expected return from today's $57.95
- **Vol: 10.2/2.3/12.6->9.5%** = GARCH says 10.2%, ATR says 2.3%, OVX says 12.6%, composite blend = 9.5%
- **65.7% (HIGH)** = model is 65.7% confident price goes UP. "HIGH" = strong conviction.
- **^ Bullish** = the model's view is bullish (up)

```
    * 95% CI = composite of GARCH(1,1) + ATR(14) + OVX implied vol
      G=GARCH | A=ATR | I=OVX implied vol | C=Composite blend
```

### Phase 6: Backtesting Results

```
  -- Backtesting --
  +----------------------------------------------------------------------------------------+
  |                       PETROQUANT -- STRATEGY PERFORMANCE REPORT                        |
  +----------------------------------------------------------------------------------------+
```

This is the full categorised performance report. Each entry has:
- **Metric name** — what's being measured
- **Value** — the number
- **Grade** — [A+]/[B]/[C]/[D-] letter grade
- **Description** — what it means in plain English

**Key takeaway from your results:**
- Total return +148% vs buy-and-hold -24% = you made money while oil lost value. Alpha = +172%!
- Sharpe 1.19 = Good risk-adjusted return
- Win rate 55.7% = You're right more often than wrong
- Max drawdown -28% = The worst dip was painful but recoverable
- Capture ratio 2.16 = You dodge losses much better than you catch gains (great for survival)

### Phase 7: Dashboard Output

```
  [OK] Dashboard rendered for: HMM + XGBoost Regime-Aware
  [OK] Saved: C:\Users\Aditya desai\Desktop\Trading Strategy\output\dashboard_hmm_xgb.html

  [OK] ALL STRATEGIES COMPLETE
    Check the /output/ folder for your interactive HTML dashboard!
```

Done! Open the HTML file in Chrome/Edge to see the full interactive 10-panel dashboard.

---

## 8. Glossary {#8-glossary}

| Term | Definition |
|------|-----------|
| **Alpha** | Extra return above the benchmark. Alpha = Strategy Return - Buy&Hold Return |
| **ATR** | Average True Range — average daily price movement over 14 days |
| **Backtest** | Testing a strategy on historical data to see how it would have performed |
| **Basis Point (bps)** | 0.01%. So 100 bps = 1%. Used for small return measurements |
| **Beta** | How much a strategy moves with the market. Beta=1 means it moves identically |
| **Binary Classifier** | AI that predicts one of two outcomes (UP or DOWN) |
| **Burke Ratio** | Return / sqrt(sum of squared drawdowns). Penalises frequent deep drops |
| **Calmar Ratio** | Annual return / max drawdown. Measures recovery ability |
| **Capture Ratio** | Up-capture / Down-capture. >1.0 = better at dodging losses than missing gains |
| **CI (Confidence Interval)** | Range where the actual value is expected to fall (95% = 19 out of 20 times) |
| **COT Report** | Commitment of Traders — weekly CFTC report showing speculator positions |
| **Crack Spread** | Refinery profit margin from turning crude oil into gasoline/diesel |
| **DXY** | US Dollar Index — measures dollar strength vs a basket of currencies |
| **Expectancy** | Average profit per trade. Must be > 0 for the strategy to be profitable long-term |
| **Feature Engineering** | Transforming raw data into patterns AI can learn from |
| **Forward-Fill (ffill)** | Fill missing values with the last known value |
| **GARCH** | Model that captures volatility clustering (big moves follow big moves) |
| **HMM** | Hidden Markov Model — detects hidden market states from price patterns |
| **Information Ratio** | Excess return / tracking error. Measures skill vs benchmark |
| **Log Return** | ln(Price_today / Price_yesterday). Better than % for math reasons |
| **Max Drawdown** | Worst peak-to-trough percentage drop in account value |
| **Momentum** | Price change over a period. Captures trend direction |
| **Omega Ratio** | Total gains / total losses. >1 = profitable |
| **OOS (Out-of-Sample)** | Data the model has never seen during training |
| **OVX** | Oil Volatility Index — the "fear gauge" for crude oil |
| **Persistence** | In GARCH: alpha + beta. Measures how "sticky" volatility is |
| **Profit Factor** | Gross profit / gross loss. >1.5 = strong edge |
| **Regime** | The market's current "mood": BULL, CHOPPY, or PANIC |
| **Regressor** | AI that predicts a continuous number (e.g., expected return +3.2%) |
| **Sharpe Ratio** | Return / volatility x sqrt(252). The gold standard risk metric |
| **Sortino Ratio** | Like Sharpe but only penalises DOWNSIDE volatility |
| **SPR** | Strategic Petroleum Reserve — US government emergency oil stockpile |
| **Sterling Ratio** | Annual return / average of top 3 drawdowns |
| **Tail Ratio** | 95th percentile gain / 5th percentile loss. >1 = big wins > big losses |
| **Treynor Ratio** | Return / beta. Rewards strategies with low market correlation |
| **Ulcer Index** | Combined depth + duration of drawdowns. Lower = less stressful |
| **Walk-Forward** | Backtesting method: train on past, test on future, slide window forward |
| **XGBoost** | Extreme Gradient Boosting — powerful tree-based machine learning algorithm |
| **Z-Score** | (Value - Mean) / Std Dev. Measures how unusual a value is |

---

> **To run PetroQuant:** `python run_strategy.py` from the project directory. Make sure `.env` has your `FRED_API_KEY` and `EIA_API_KEY`. The HTML dashboard will appear in `/output/`.
