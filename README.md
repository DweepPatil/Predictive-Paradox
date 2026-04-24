# ⚡ Power Demand Forecasting

A production-grade, end-to-end machine learning pipeline for **next-hour electricity demand forecasting** on the national grid. Built with LightGBM on a fully engineered multi-source dataset spanning 2015–2025.

---

## 🔧 Data Preprocessing Pipeline (`Data_Preprocessing.ipynb`)

### Phase 1 — Data Ingestion & Initial Cleaning

Three raw datasets (power grid, weather, macroeconomic) are loaded and aligned. The power dataset undergoes:
- **Deduplication** on the `datetime` column, keeping the last (most-corrected) reading.
- **Column subsetting** from 15 down to 3 essential grid metrics: `generation_mw`, `demand_mw`, `load_shedding`.
- **Strict hourly resampling** via `resample('h').mean()`, which averages sub-hourly readings and exposes missing hours as explicit `NaN` rows.

---

### Phase 2 — Advanced Outlier Neutralization (Grid Physics)

A centralized `neutralize_all_outliers` function handles anomalies in three ordered passes:

**Pass 1 — Domain Knowledge (Load Shedding Threshold):**
Any `load_shedding` value exceeding **4,000 MW** is physically impossible on the grid. These spikes contaminate the entire hour's reading, so `generation_mw`, `demand_mw`, and `load_shedding` are all blanked to `NaN` simultaneously for those corrupt rows.

**Pass 2 — Statistical Limits (IQR Method, 3.0× multiplier):**
After domain-knowledge cleaning, the Interquartile Range with a 3× multiplier flags mathematically impossible spikes in `generation_mw` and `demand_mw`. A 3.0× (vs. the standard 1.5×) multiplier is intentionally conservative — it targets only genuine anomalies rather than legitimate peak-demand events.

**Pass 3 — Bivariate Cross-Validation (Grid Physics):**
A physical constraint is enforced: `expected_demand = generation_mw + load_shedding`. Any hour where actual `demand_mw` deviates from this physical expectation by more than **5%** is classified as a bivariate outlier — a reading internally inconsistent with grid physics — and neutralized to `NaN`.

---

### Phase 3 — Strategic Missing Value Imputation

Missing gaps are measured and classified into three tiers before any filling is applied:

| Gap Tier | Duration | Treatment |
|---|---|---|
| **Micro gaps** | ≤ 6 hours | Linear interpolation — smoothly bridges short, localized outages |
| **Medium gaps** | 7 – 336 hours (up to 14 days) | **Blended history imputation** — averages the identical hour from 1 week ago and 2 weeks ago, preserving weekly behavioral cycles |
| **Black holes** | > 336 hours (> 14 days) | Left as `NaN` — too dangerous to fabricate; passed to the model to handle implicitly |

A secondary polishing pass targets residual micro-gaps (≤ 10 hours) that survived because their historical reference weeks also contained missing data — termed *historical shadows*.

---

### Phase 4 — Temporal & Target-Aligned Feature Engineering

#### Temporal Features Engineered

| Feature | Why It Was Engineered |
|---|---|
| `hour`, `month`, `year` | Ordinal anchors for intra-day, seasonal, and long-term structural trends |
| `is_weekend` | Binary flag — industrial and commercial load drops significantly on weekends |
| `season` | Quarterly climate classification (monsoon, winter, pre-summer, summer) |
| `hour_sin`, `hour_cos` | Cyclical encoding — ensures the model understands that hour 23 flows into hour 0, not into infinity |
| `month_sin`, `month_cos` | Cyclical encoding for continuous seasonal transitions |
| `day_of_week_sin`, `day_of_week_cos` | Cyclical encoding for smooth Sunday→Monday boundary |

#### Lag & Momentum Features Engineered

| Feature | Window | Why It Was Engineered |
|---|---|---|
| `lag_previous_hour` | shift(1) | Immediate momentum — the most recent grid state baseline |
| `lag_yesterday_target_hour` | shift(23) | Same hour, yesterday — captures daily behavioral patterns |
| `lag_lastweek_target_hour` | shift(167) | Same hour, last week — captures full weekly behavioral cycles |
| `rolling_mean_6h` | 6-hour window | Short-term momentum smoothing |
| `rolling_mean_24h` | 24-hour window | Daily momentum context |
| `target_next_hour_demand` | shift(-1) | Supervised learning target — created without leakage by shifting demand backward |

---

### Phase 5 — Weather Data Processing

The weather dataset is resampled to strict 1-hour frequency, followed by a **physical boundary audit** to check sensor integrity — temperature readings outside −5°C to 55°C, precipitation outside 0–150 mm/hr, and cloud cover outside 0–100% are all flagged as impossible sensor readings.

Only three meteorological features are retained based on domain knowledge of their influence on electricity demand: `apparent_temperature`, `precipitation_mm`, and `cloud_cover_pct`.

---

### Phase 6 — Macroeconomic Data Structuring (Leakage Prevention)

Three World Bank indicators are selected — **GDP**, **Manufacturing Value Added**, and **Population Total** — and transformed from wide-matrix format into a time-series-compatible long format via `melt` → `pivot`.

**Look-ahead bias prevention:** Each year's economic data is shifted forward by **one full year** (e.g., 2014 data only becomes accessible on 2015-01-01), enforcing a strict publication-lag safeguard. A forward-fill (`ffill`) bridges the unpublished 2023–2024 period.

---

### Phase 7 — Master Assembly

- **Grid + Weather** are joined via an inner merge on the datetime index, retaining only hours with complete physical and environmental records.
- **Macroeconomic data** is integrated using `pd.merge_asof(direction='backward')` — a strict time-series safeguard guaranteeing each hourly row joins only the most recently published economic data relative to its timestamp, making the final dataset mathematically leak-proof.

The final master dataset contains **27 columns** and is exported as `Final_dataset.csv`.

---

## 🤖 Model (`Model.ipynb`)

### Algorithm: LightGBM Regressor

LightGBM was chosen for its highly optimized gradient-boosted tree architecture, native handling of missing values (the black-hole `NaN` periods), and fast training on large tabular datasets.

**Hyperparameters:**

| Parameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 500 | Sufficient ensemble depth for complex temporal patterns |
| `learning_rate` | 0.05 | Conservative rate for stable gradient descent |
| `max_depth` | 8 | Limits tree depth to prevent overfitting |
| `random_state` | 42 | Reproducibility |
| `n_jobs` | -1 | Full parallel CPU utilization |

---

### Feature Selection

Two features were intentionally excluded after iterative testing:

- **`generation_mw`** — Near-identical to `demand_mw` (offset only by load shedding), creating heavy collinearity that dilutes signal.
- **`day_of_week`** (raw categorical) — Acts as a lazy shortcut. Removing it forces the model to navigate the week using the mathematically correct `day_of_week_sin` / `cos` cyclical encodings, producing smoother boundary transitions.

**Total features fed to the model: 24**

---

### Chronological Train/Test Split

| Split | Period |
|---|---|
| **Training** | 2015 – 2023-12-31 |
| **Test (Hold-out)** | 2024-01-01 – 2025-06-17 |

The split is strictly chronological — the model never sees any future data during training.

---

## 📊 Evaluation & Results

Performance is evaluated on the chronological hold-out test set using **Mean Absolute Percentage Error (MAPE)**.

| Model | Test MAPE |
|---|---|
| **Naive Baseline** (previous hour = next hour) | 3.33% |
| **LightGBM** | **1.75%** |

### Interpreting the MAPE Magnitude

The Naive Baseline MAPE of **3.33%** reflects the natural intra-hour volatility of the grid — demand changes fast enough that simply repeating the current reading fails by over 3% on average.

The LightGBM MAPE of **1.75%** represents a **47.4% error reduction** over the naive baseline, meaning the model cuts prediction error nearly in half by learning temporal geometry, behavioral cycles, and weather coupling. For a national grid context — where even 1% of demand can represent hundreds of megawatts — a 1.75% MAPE is operationally significant. Predictions are within 1.75% of actual demand on average, across an **18-month future hold-out window** the model has never seen.

---

## 🧠 Feature Importance Analysis

LightGBM's `feature_importances_` counts how many times each feature was used as a decision split across all 500 trees. The chart reveals a clear and physically logical four-tier hierarchy:

### Tier 1 — Dominant Driver: Current Grid State

**`demand_mw`** is the single most important feature by a wide margin (~2,050 splits) — nearly 30% more than the next feature. This confirms that current grid load is the strongest predictor of next-hour demand, as electricity consumption is highly autocorrelated at the 1-hour lag. This also retroactively validates the decision to drop `generation_mw`: with collinearity removed, `demand_mw` cleanly absorbed the full baseline signal.

### Tier 2 — Temporal Geometry (~750–1,575 splits)

**`hour`**, **`hour_sin`**, **`hour_cos`**, **`month`**, and **`rolling_mean_24h`** cluster as the second tier. Notably, both the raw `hour` (ordinal) and its cyclical encodings (`hour_sin`, `hour_cos`) rank highly — the model uses the ordinal form for absolute thresholds ("is it past 6 PM?") and the sine/cosine form for smooth cycle-boundary transitions. This dual-use confirms that cyclical encoding added genuine value rather than creating redundancy.

### Tier 3 — Behavioral Memory & Weather (~850–1,225 splits)

**`lag_yesterday_target_hour`** and **`lag_lastweek_target_hour`** both rank *higher* than `lag_previous_hour`. This is the most revealing insight from the chart: the model finds that *what happened at this exact hour yesterday and last week* is more predictive than the immediate prior hour. It means human behavioral patterns — work schedules, sleep cycles, weekly industrial rhythms — are stronger demand drivers than short-term momentum alone. **`apparent_temperature`** also sits in this tier (~975 splits), confirming strong weather-demand coupling driven by air conditioning load.

### Tier 4 — Low Signal / Structural Features (near-zero splits)

**`manufacturing_constant_2015_usd`**, **`gdp_constant_2015_usd`**, and **`population_total`** all register effectively zero splits. This does not mean they are useless — these macroeconomic features provide the long-run structural baseline that keeps the model calibrated across multi-year growth trends in the dataset. However, LightGBM correctly deprioritized them because annual indicators change too slowly to add discriminative power at the hourly split level. Their near-zero importance confirms the model is not overfitting to economic noise.

---

## ⚙️ How to Run

**1. Prerequisites**
```bash
pip install pandas numpy matplotlib seaborn lightgbm scikit-learn openpyxl
```

**2. Data Preprocessing**

Run all cells in `Data_Preprocessing.ipynb`. This executes the full 7-phase cleaning and engineering pipeline and outputs `./Dataset/Final_dataset.csv`.

**3. Model Training & Evaluation**

Run all cells in `Model.ipynb`. This trains LightGBM on 2015–2023 data, evaluates on the 2024–2025 hold-out set, and displays the MAPE comparison and feature importance chart.

---

## 🛡️ Key Design Principles

| Principle | Implementation |
|---|---|
| **No time travel** | Strict chronological train/test split; `merge_asof(direction='backward')` for macro data |
| **No leakage** | Economic data shifted forward 1 year; target created via `shift(-1)` |
| **Physics-first cleaning** | Domain knowledge (4,000 MW shedding threshold, 5% grid imbalance tolerance) applied before statistical methods |
| **Black holes preserved** | Gaps > 14 days left as `NaN` — not fabricated — to avoid poisoning the model with guessed data |
| **Cyclical encoding** | Sine/cosine transforms on all periodic time features prevent artificial discontinuities at cycle boundaries |
