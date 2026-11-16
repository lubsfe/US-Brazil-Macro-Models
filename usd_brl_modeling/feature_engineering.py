import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from config import PANEL_START_DATE


def build_monthly_macro_panel(
        raw_data: dict[str, pd.DataFrame],
        panel_start: str = PANEL_START_DATE,
) -> pd.DataFrame:
    """Aligns daily and monthly series to month-end, engineers macro factors, and avoids look-ahead bias."""
    daily_keys = ["usd_brl", "fed_funds", "selic_target", "vix"]
    monthly_parts: list[pd.Series] = []

    for key in daily_keys:
        s = (
            raw_data[key]
            .set_index("date")[key]
            .resample("ME")
            .last()
        )
        monthly_parts.append(s)

    monthly_keys = ["us_cpi", "br_ipca_mom", "commodities"]
    for key in monthly_keys:
        df_m = raw_data[key].copy()
        df_m["date"] = df_m["date"].dt.to_period("M").dt.to_timestamp("M")
        s = df_m.drop_duplicates(subset=["date"]).set_index("date")[key]
        monthly_parts.append(s)

    panel = pd.concat(monthly_parts, axis=1, join="inner").sort_index()

    # 12-month inflation rates using the 2014 lookback buffer
    panel["us_cpi_yoy"] = (panel["us_cpi"] / panel["us_cpi"].shift(12) - 1.0) * 100.0
    panel["br_ipca_yoy"] = (
                                   (1.0 + panel["br_ipca_mom"] / 100.0)
                                   .rolling(window=12)
                                   .apply(np.prod, raw=True)
                                   - 1.0
                           ) * 100.0

    # Nominal & real carry spreads, inflation differential, and external shocks
    panel["rate_diff"] = panel["selic_target"] - panel["fed_funds"]
    panel["delta_rate_diff"] = panel["rate_diff"].diff()
    panel["inflation_diff"] = panel["br_ipca_yoy"] - panel["us_cpi_yoy"]

    panel["us_real_rate"] = panel["fed_funds"] - panel["us_cpi_yoy"]
    panel["br_real_rate"] = panel["selic_target"] - panel["br_ipca_yoy"]
    panel["real_rate_diff"] = panel["br_real_rate"] - panel["us_real_rate"]
    panel["delta_real_rate_diff"] = panel["real_rate_diff"].diff()

    # Percentage log returns (x100) and risk appetite shifts
    panel["log_fx_ret"] = np.log(panel["usd_brl"] / panel["usd_brl"].shift(1)) * 100.0
    panel["log_fx_ret_lag1"] = panel["log_fx_ret"].shift(1)
    panel["log_comm_ret"] = np.log(panel["commodities"] / panel["commodities"].shift(1)) * 100.0
    panel["delta_vix"] = panel["vix"].diff()

    # Publication-lagged macro predictors (known at month-end t for predicting t -> t+1)
    panel["real_rate_diff_lag1"] = panel["real_rate_diff"].shift(1)
    panel["delta_real_rate_diff_lag1"] = panel["delta_real_rate_diff"].shift(1)
    panel["inflation_diff_lag1"] = panel["inflation_diff"].shift(1)
    panel["log_comm_ret_lag1"] = panel["log_comm_ret"].shift(1)

    # Forward 1-month target return and realized next-month spot rate
    panel["target_fx_ret_1m"] = panel["log_fx_ret"].shift(-1)
    panel["target_usd_brl_1m"] = panel["usd_brl"].shift(-1)

    panel = panel.loc[panel.index >= pd.to_datetime(panel_start)].copy()
    panel.index.name = "date"
    return panel


def run_adf_stationarity_tests(panel: pd.DataFrame) -> pd.DataFrame:
    """Runs Augmented Dickey-Fuller tests on levels vs. differenced/return transformations."""
    test_specs = [
        ("usd_brl", "Level (Spot BRL/USD)"),
        ("log_fx_ret", "Log Return (% MoM)"),
        ("rate_diff", "Level (Selic - Fed Funds)"),
        ("delta_rate_diff", "First Difference (pp)"),
        ("real_rate_diff", "Level (Real Carry Diff)"),
        ("delta_real_rate_diff", "First Difference (pp)"),
        ("commodities", "Level (IMF Index)"),
        ("log_comm_ret", "Log Return (% MoM)"),
        ("vix", "Level (Index)"),
        ("delta_vix", "First Difference (pts)"),
    ]

    results = []
    for col, form_label in test_specs:
        series = panel[col].dropna()
        adf_stat, p_val, used_lag, _, _, _ = adfuller(
            series, autolag="AIC", result_object=False
        )
        results.append({
            "variable": col,
            "transformation": form_label,
            "adf_stat": round(float(adf_stat), 4),
            "p_value": round(float(p_val), 4),
            "lags": int(used_lag),
            "stationary_5pct": "Yes" if p_val < 0.05 else "No (Unit Root)",
        })
    return pd.DataFrame(results)