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

Through rigorous iterative testing, a massive **11 features** were intentionally pruned from the final model. Removing this data actually *increased* the AI's predictive intelligence by eliminating collinearity, lazy categorical shortcuts, and high-frequency noise:

- **Collinear Baselines:** `generation_mw` was dropped because it is near-identical to `demand_mw`. Feeding the model both dilutes the primary signal.
- **Categorical Crutches:** `day_of_week` and `is_weekend` were dropped. Removing these rigid boxes forces the model to navigate the week using the mathematically continuous `day_of_week_sin` / `cos` cyclical encodings, producing smoother boundary transitions.
- **Point-in-Time Lags:** `lag_previous_hour`, `lag_yesterday_target_hour`, and `lag_lastweek_target_hour` were dropped. Exact historical lags introduce an "echo of noise" (e.g., blindly predicting a spike today just because a random anomaly happened exactly 24 hours ago). Removing these forces the model to rely purely on the *smoothed momentum* of rolling averages.
- **Meteorological Noise:** `precipitation_mm` and `cloud_cover_pct` were dropped, isolating `apparent_temperature` as the sole, undisputed thermodynamic driver of human cooling/heating load.
- **Macroeconomic Collinearity:** `year`, `manufacturing_constant_2015_usd`, and `population_total` were dropped. Because these broadly trend upward together, they create collinear noise. Leaving *only* `gdp_constant_2015_usd` provides the AI with a single, perfectly clean structural anchor for the grid's multi-year growth.

**Total features fed to the final model: 15**

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
| **LightGBM (Highly Tuned)** | **1.71%** |

### Interpreting the MAPE Magnitude

The Naive Baseline MAPE of **3.33%** reflects the natural intra-hour volatility of the grid — demand changes fast enough that simply repeating the current reading fails by over 3% on average.

The fully pruned LightGBM model's MAPE of **1.71%** represents an incredible **48.6% error reduction** over the naive baseline. By successfully stripping away noisy lags and redundant variables, the model cuts prediction error nearly in half, relying purely on smoothed momentum, temporal geometry, and temperature. For a national grid context, predictions within 1.71% across an **18-month future hold-out window** are operationally superb.

---

## 🧠 Feature Importance Analysis

LightGBM's `feature_importances_` counts how many times each feature was used as a decision split across all 500 trees. With all noise, exact lags, and collinearity removed, the AI's "brain" reveals a perfectly logical hierarchy:

### Tier 1 — Dominant Driver: Immediate Grid State
**`demand_mw`** is the undisputed king (nearly 2,500 splits). With historical lags gone, the model leans heavily on the immediate prior state of the grid. This confirms electricity consumption is highly autocorrelated at the 1-hour mark, retroactively validating the removal of `generation_mw` to consolidate the baseline signal.

### Tier 2 — Daily Geometry
**`hour`** and **`hour_sin`** form the second tier. The model calculates the context of the demand by using the ordinal `hour` for absolute thresholds (e.g., "is the sun down?") and the continuous `hour_sin` to mathematically map the smooth human behavioral cycle of the day.

### Tier 3 — Weather & Smoothed Momentum
**`apparent_temperature`**, **`rolling_mean_24h`**, and **`rolling_mean_6h`** cluster together here. Stripped of point-in-time lags, the AI uses rolling means to understand recent grid momentum without overfitting to yesterday's micro-spikes. It combines this momentum directly with the apparent temperature to calculate how cooling/heating loads will warp the baseline schedule.

### Tier 4 — Macroscopic Anchors
**`month`**, **`gdp_constant_2015_usd`**, and **`hour_cos`** sit in the 750–900 split range. A massive success of the feature pruning is visible here: by dropping the redundant Population, Manufacturing, and Year columns, **GDP** cleanly absorbed the entire macroeconomic signal. The AI uses it exactly as intended—as an annual structural baseline to keep the model calibrated across multi-year economic growth.

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
