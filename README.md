# US-Brazil Macroeconomic Regression Analysis

## Executive Summary
This project models the direct impact of US monetary policy on Brazilian interest rates using an Ordinary Least Squares (OLS) regression. By programmatically retrieving and merging 108 months of macroeconomic data via the FRED and BCB APIs, this analysis quantifies the relationship between the US Federal Funds Rate and the Brazilian Selic Rate.

**Key Findings:**
* **Strong US Influence:** The US Federal Funds Rate dictates approximately 58% of the historical variance in the Brazilian Selic rate ($R^2 = 0.584$).
* **Quantifiable Impact:** For every 100 basis point (1%) increase by the US Federal Reserve, the Banco Central do Brasil responds with an average rate hike of 64 basis points to mitigate capital flight and currency depreciation.
* **Statistical Significance:** The modeled relationship is highly significant ($p < 0.001$), underscoring the tight financial linkage between US and Brazilian debt markets.

## Data Sources
* **Federal Reserve Economic Data (FRED):** US Federal Funds Effective Rate (Monthly)
* **Banco Central do Brasil (BCB):** Selic Target Rate (Aggregated from daily observations to monthly averages)

## Methodology and Limitations
The data was cleaned and merged using `pandas`, ensuring proper time-series alignment before running an OLS regression using `statsmodels`.

**Note on Autocorrelation:** The Durbin-Watson statistic (0.028) indicates strong positive time-series autocorrelation. Future iterations of this model will employ first-differencing or an ARIMA model to account for the dependency of current interest rates on past rates.

## How to Run
1. Clone the repository.
2. Install dependencies: `pip install pandas requests statsmodels`
3. Add your FRED API key to the script.
4. Run `fedfunds_selic_regression.py` to fetch current data and generate the OLS summary.
