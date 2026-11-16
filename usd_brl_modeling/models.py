import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import PREDICTIVE_FEATURES


def fit_hac_ols_regression(
        panel: pd.DataFrame,
        target_col: str,
        feature_cols: list[str],
        expected_signs: dict[str, str],
        hac_lags: int = 3,
) -> tuple[sm.regression.linear_model.RegressionResultsWrapper, pd.DataFrame]:
    """Fits an OLS regression with Newey-West HAC standard errors and formats diagnostics."""
    reg_df = panel[[target_col, *feature_cols]].dropna().copy()
    y = reg_df[target_col]
    X = sm.add_constant(reg_df[feature_cols])

    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})

    rows = []
    for param_name in model.params.index:
        coef = float(model.params[param_name])
        se = float(model.bse[param_name])
        t_stat = float(model.tvalues[param_name])
        p_val = float(model.pvalues[param_name])
        obs_sign = "+" if coef > 0 else "-"
        exp_sign = expected_signs.get(param_name, "n/a")

        rows.append({
            "term": param_name,
            "coef": round(coef, 4),
            "hac_se": round(se, 4),
            "t_stat": round(t_stat, 3),
            "p_value": round(p_val, 4),
            "expected_sign": exp_sign,
            "actual_sign": obs_sign,
        })

    summary_table = pd.DataFrame(rows)
    return model, summary_table


def run_econometric_baselines(panel: pd.DataFrame) -> dict[str, object]:
    """Estimates contemporaneous factor attribution and 1-month-ahead predictive HAC OLS models."""
    contemp_features = [
        "delta_real_rate_diff",
        "log_comm_ret",
        "delta_vix",
        "log_fx_ret_lag1",
    ]
    contemp_signs = {
        "const": "n/a",
        "delta_real_rate_diff": "-",
        "log_comm_ret": "-",
        "delta_vix": "+",
        "log_fx_ret_lag1": "+/-",
    }
    model_contemp, table_contemp = fit_hac_ols_regression(
        panel=panel,
        target_col="log_fx_ret",
        feature_cols=contemp_features,
        expected_signs=contemp_signs,
        hac_lags=3,
    )

    pred_signs = {
        "const": "n/a",
        "delta_real_rate_diff_lag1": "-",
        "delta_rate_diff": "-",
        "log_comm_ret_lag1": "-",
        "delta_vix": "+",
        "log_fx_ret": "+/-",
    }
    model_pred, table_pred = fit_hac_ols_regression(
        panel=panel,
        target_col="target_fx_ret_1m",
        feature_cols=PREDICTIVE_FEATURES,
        expected_signs=pred_signs,
        hac_lags=3,
    )

    return {
        "contemp_model": model_contemp,
        "contemp_table": table_contemp,
        "pred_model": model_pred,
        "pred_table": table_pred,
    }


def run_walk_forward_validation(
        panel: pd.DataFrame,
        feature_cols: list[str] = PREDICTIVE_FEATURES,
        min_train_months: int = 60,
) -> dict[str, pd.DataFrame]:
    """Evaluates expanding-window out-of-sample forecasts against the Meese-Rogoff Random Walk."""
    req_cols = ["usd_brl", "target_fx_ret_1m", "target_usd_brl_1m", *feature_cols]
    wf_df = panel[req_cols].dropna().copy()

    model_specs = {
        "OLS_Linear": lambda: LinearRegression(),
        "Ridge_L2": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "Random_Forest": lambda: RandomForestRegressor(
            n_estimators=200,
            max_depth=3,
            min_samples_leaf=5,
            random_state=42,
        ),
        "Gradient_Boosting": lambda: GradientBoostingRegressor(
            n_estimators=60,
            max_depth=2,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        ),
    }

    oos_records: list[dict] = []
    last_ridge_pipe = None

    for t in range(min_train_months, len(wf_df)):
        train_slice = wf_df.iloc[:t]
        test_row = wf_df.iloc[[t]]

        X_train = train_slice[feature_cols]
        y_train = train_slice["target_fx_ret_1m"]
        X_test = test_row[feature_cols]

        spot_t = float(test_row["usd_brl"].iloc[0])
        actual_ret = float(test_row["target_fx_ret_1m"].iloc[0])
        actual_spot_next = float(test_row["target_usd_brl_1m"].iloc[0])

        row_out = {
            "feature_date": test_row.index[0],
            "forecast_date": test_row.index[0] + pd.offsets.MonthEnd(1),
            "spot_t": spot_t,
            "actual_ret_pct": actual_ret,
            "actual_spot_next": actual_spot_next,
            "pred_ret_Random_Walk": 0.0,
            "pred_spot_Random_Walk": spot_t,
        }

        for name, builder in model_specs.items():
            est = builder()
            est.fit(X_train, y_train)
            pred_ret = float(est.predict(X_test)[0])
            row_out[f"pred_ret_{name}"] = pred_ret
            row_out[f"pred_spot_{name}"] = spot_t * float(np.exp(pred_ret / 100.0))

            if name == "Ridge_L2":
                last_ridge_pipe = est

        oos_records.append(row_out)

    oos_df = pd.DataFrame(oos_records)

    rw_ret_err = oos_df["actual_ret_pct"] - oos_df["pred_ret_Random_Walk"]
    rw_ret_rmse = float(np.sqrt(np.mean(rw_ret_err ** 2)))

    all_model_names = ["Random_Walk", *model_specs.keys()]
    metric_rows = []

    for name in all_model_names:
        pred_ret = oos_df[f"pred_ret_{name}"]
        pred_spot = oos_df[f"pred_spot_{name}"]

        ret_rmse = float(np.sqrt(np.mean((oos_df["actual_ret_pct"] - pred_ret) ** 2)))
        spot_rmse = float(np.sqrt(np.mean((oos_df["actual_spot_next"] - pred_spot) ** 2)))
        spot_mae = float(np.mean(np.abs(oos_df["actual_spot_next"] - pred_spot)))
        theils_u = ret_rmse / rw_ret_rmse

        if name == "Random_Walk":
            dir_acc_str = "n/a (0-drift)"
        else:
            non_zero = oos_df["actual_ret_pct"] != 0
            hits = np.sign(pred_ret[non_zero]) == np.sign(oos_df.loc[non_zero, "actual_ret_pct"])
            dir_acc_str = f"{float(hits.mean()) * 100.0:.1f}%"

        metric_rows.append({
            "model": name,
            "oos_months": len(oos_df),
            "ret_rmse_pct": round(ret_rmse, 4),
            "spot_rmse_brl": round(spot_rmse, 4),
            "spot_mae_brl": round(spot_mae, 4),
            "theils_u_vs_rw": round(theils_u, 4),
            "dir_accuracy": dir_acc_str,
        })

    metrics_df = pd.DataFrame(metric_rows)

    ridge_coefs = last_ridge_pipe.named_steps["ridge"].coef_
    weights_df = (
        pd.DataFrame({
            "feature": feature_cols,
            "std_ridge_coef": np.round(ridge_coefs, 4),
            "abs_weight": np.round(np.abs(ridge_coefs), 4),
        })
        .sort_values("abs_weight", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "oos_predictions": oos_df,
        "metrics": metrics_df,
        "ridge_weights": weights_df,
    }