# USD/BRL Exchange Rate Modeling & Walk-Forward Macro Backtest

An end-to-end econometric and machine learning pipeline analyzing what drives the **USD/BRL** exchange rate (2015–2026) and testing how well macroeconomic fundamentals hold up out of sample. Built entirely on live data pulled directly from the **Federal Reserve Economic Data (FRED)** API and **Banco Central do Brasil (BCB) SGS** REST API.

---

## 1. Why I Built This
Exchange rates sit right at the intersection of economics and data science—and they are notoriously humbling to model. In macroeconomics courses, we learn that currency movements should reflect interest rate differentials (carry), inflation gaps (purchasing power parity), terms-of-trade shifts (commodity prices), and global risk appetite. Yet in practice, floating emerging-market currencies like the Brazilian Real (`BRL`) are whipsawed by fiscal headlines, central bank credibility shifts, and global liquidity cycles.

I built this project to test two questions empirically:
1. **Within-Month Attribution:** How much of USD/BRL's monthly movement can be explained contemporaneously by changes in real carry, global commodities, and market volatility (`VIX`)?
2. **Out-of-Sample Predictability:** Using strictly look-ahead-free predictors in a realistic walk-forward backtest, can regularized linear or tree-based ML models beat the driftless **Meese–Rogoff Random Walk** ($\hat{S}_{t+1} = S_t$)—the classic benchmark in international finance?

---

## 2. What Makes This Project Different
Many student FX notebooks rely on static Kaggle CSVs, run regressions on non-stationary price levels, or use a random `train_test_split` that leaks future macro regimes into the training set. This pipeline was designed to avoid those traps:

* **Zero Static Datasets — Live Cross-Border API Ingestion:** Every run pulls raw daily and monthly series straight from the Federal Reserve and Banco Central do Brasil. The ingestion layer handles real-world data engineering friction automatically, including FRED's `"."` market-holiday markers and BCB's 10-year query cap on daily series (by chunking requests into 8-year windows).
* **Strict Publication-Lag & Anti-Leakage Discipline:** At month-end $t$, spot FX, VIX, and policy rates are observable in real time, whereas US CPI, Brazil IPCA, and IMF commodity indices for month $t$ are not published until month $t+1$. Predictive models lag published macro releases by one full month, and `StandardScaler` is fit strictly inside each expanding training window so future volatility never leaks backward.
* **Econometrics + ML Side by Side:** Rather than treating machine learning as a black box, the workflow starts with Augmented Dickey-Fuller (ADF) stationarity diagnostics and Newey–West HAC regressions to verify economic signs before testing expanding-window Ridge, OLS, Random Forest, and Gradient Boosting models against a zero-drift Random Walk.

---

## 3. Data Pipeline & Feature Engineering
All series are fetched starting in `2014-01-01`—providing a 12-month lookback buffer so year-over-year inflation rates are fully populated starting January 2015—and aligned to month-end (`ME`) closing snapshots:

| Series / Feature | Source & ID | Engineering & Economic Role |
| :--- | :--- | :--- |
| **Spot USD/BRL** | FRED `DEXBZUS` | Month-end close; monthly log return $\Delta s_t = 100 \times \ln(S_t / S_{t-1})$ |
| **US Effective Fed Funds** | FRED `DFF` | Month-end daily rate (% p.a.) |
| **Brazil Selic Target** | BCB SGS `432` | Month-end COPOM target (% p.a.) |
| **US CPI (All Urban)** | FRED `CPIAUCSL` | 12-month YoY % inflation (lagged 1M in predictive models) |
| **Brazil IPCA (Monthly %)** | BCB SGS `433` | Rolling 12-month compounded YoY % inflation (lagged 1M in predictive models) |
| **Global Risk Appetite** | FRED `VIXCLS` | Month-end CBOE VIX close and monthly first difference ($\Delta \text{VIX}_t$) |
| **Global Commodities Index** | FRED `PALLFNFINDEXM` | IMF All Commodities monthly log return (lagged 1M in predictive models) |

### Derived Carry Spreads
* **Nominal Carry Differential:** $\text{rate\_diff}_t = \text{Selic}_t - \text{FedFunds}_t$
* **Ex-Post Real Carry Differential:** $\text{real\_rate\_diff}_t = (\text{Selic}_t - \text{IPCA}_{t}^{\text{YoY}}) - (\text{FedFunds}_t - \text{CPI}_{t}^{\text{YoY}})$

---

## 4. Empirical Results

### Stationarity Diagnostics (Augmented Dickey-Fuller Tests)
Testing levels against transformations confirmed that raw spot `usd_brl` ($p = 0.3218$), `real_rate_diff` ($p = 0.3915$), and `commodities` ($p = 0.5959$) all contain unit roots ($I(1)$). Transforming variables into monthly log returns and first differences achieved stationarity across the board ($p < 0.03$), preventing spurious level regressions.

### Newey–West HAC Regressions (`maxlags=3`)
* **Model A — Contemporaneous Attribution ($\Delta s_t$, $N = 139$, $R^2 = 0.1454$, $F = 6.189$, $p = 0.0001$):** External global shocks drive the bulk of within-month variance. A $+1\%$ rally in global commodities strengthens the Real by **$-0.24\%$** ($t = -2.64$, $p = 0.0084$), while a $+1\text{ pt}$ spike in the VIX weakens it by **$+0.16\%$** ($t = +2.78$, $p = 0.0054$).
* **Model B — 1-Month-Ahead Predictive Baseline ($\Delta s_{t+1}$, $N = 137$, $R^2 = 0.0457$):** Once lagged to respect macro release schedules, a $+1\text{ pp}$ widening in Brazil's real carry advantage (`delta_real_rate_diff_lag1`, $\beta = -1.0846$, $t = -2.21$, $p = 0.0274$) predicts BRL appreciation in the following month, whereas commodity shocks are already absorbed contemporaneously.

### Expanding-Window Walk-Forward Backtest (77 Months: Apr 2020 – Sep 2026)
Trained on a minimum 60-month initial window (2015–2020) and stepped forward one month at a time:

| Model | Return RMSE (%) | Spot RMSE (BRL) | Spot MAE (BRL) | Theil's $U$ vs. RW | Directional Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Walk (0-Drift)** | **3.4846%** | **R$ 0.1843** | **R$ 0.1509** | **1.0000** | n/a |
| **Ridge Regression (L2)** | 3.8614% | R$ 0.2048 | R$ 0.1585 | 1.1081 | **55.8%** |
| **OLS Linear** | 3.9617% | R$ 0.2101 | R$ 0.1612 | 1.1369 | 53.2% |
| **Random Forest** | 3.8231% | R$ 0.2036 | R$ 0.1647 | 1.0972 | 45.5% |
| **Gradient Boosting** | 4.0228% | R$ 0.2147 | R$ 0.1773 | 1.1544 | 40.3% |

---

## 5. What I Learned (Economics & Data Science Takeaways)

1. **Two-Speed Macro Transmission is Real:** Comparing Model A and Model B made an important economic distinction tangible. Global risk-off episodes (`VIX`) and commodity price swings hit emerging-market currencies almost immediately within the same month, whereas domestic monetary policy credibility—captured by shifts in the ex-post real interest rate spread—transmits into the exchange rate with a lag.
2. **In Small-Sample Macro Data, Regularization Beats Complexity:** With ~140 monthly observations, high-capacity tree ensembles (`Random Forest` and `Gradient Boosting`) overfit noise in the training window, dropping to `45.5%` and `40.3%` directional accuracy out of sample. By contrast, L2-regularized `Ridge` regression shrunk noisy coefficients, improved spot RMSE over unregularized OLS (`R$ 0.2048` vs. `R$ 0.2101`), and preserved a **55.8% directional hit rate**.
3. **Why the Meese–Rogoff Puzzle Still Endures:** Every parametric and ML model recorded a Theil's $U > 1.00$ on point magnitude RMSE relative to the driftless Random Walk. Seeing that play out empirically reinforced why short-horizon FX forecasting is so difficult: even when macro fundamentals point in the right direction more often than not, unanticipated geopolitical and fiscal shocks dominate the *size* of monthly exchange rate moves.
4. **Domain Context Matters When Integrating Cross-Country APIs:** Merging US and Brazilian macro data isn't just a `pd.merge` exercise. Understanding that FRED reports US CPI as a price level index (`CPIAUCSL`) while BCB reports IPCA as a monthly percentage variation (`SGS 433`)—requiring rolling 12-month geometric compounding—was essential to constructing an apples-to-apples real carry spread.

---

## 6. Repository Structure & Usage
* `usd_brl_model.py`: Main pipeline orchestrator executing all five workflow steps end to end.
* `config.py`: Shared directory paths, sample date boundaries, and predictive feature definitions.
* `data_ingestion.py`: Live FRED and Banco Central do Brasil (BCB SGS) REST API fetchers with 8-year chunking.
* `feature_engineering.py`: Month-end frequency alignment, YoY inflation compounding, carry spread engineering, and ADF stationarity tests.
* `models.py`: Newey–West HAC econometric factor regressions and expanding-window walk-forward ML backtesting.
* `visualization.py`: Dataset export and multi-panel diagnostic figure generation.
* `usd_brl_macro_dataset.csv`: Exported monthly panel (`2015-01-31` to `2026-09-30`) containing raw series, engineered features, and aligned out-of-sample forecasts.
* `usd_brl_drivers_and_forecast.png`: Three-tier diagnostic chart visualizing spot USD/BRL vs. walk-forward Ridge forecasts, nominal vs. real carry spreads, standardized Ridge feature weights, and out-of-sample RMSE benchmarks.

```bash
python usd_brl_modeling/usd_brl_model.py