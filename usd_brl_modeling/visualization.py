from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd

from config import PROJECT_DIR


def export_artifacts(
        panel: pd.DataFrame,
        wf_results: dict[str, pd.DataFrame],
        output_dir: Path = PROJECT_DIR,
) -> None:
    """Exports the engineered macro panel + OOS predictions to CSV and renders a 3-tier summary figure."""
    csv_path = output_dir / "usd_brl_macro_dataset.csv"
    chart_path = output_dir / "usd_brl_drivers_and_forecast.png"

    oos_df = wf_results["oos_predictions"].copy()
    metrics_df = wf_results["metrics"].copy()
    weights_df = wf_results["ridge_weights"].copy()

    oos_merge = oos_df.set_index("forecast_date")[[
        "pred_ret_Ridge_L2",
        "pred_spot_Ridge_L2",
        "pred_spot_Random_Walk",
        "pred_spot_Random_Forest",
    ]]
    export_df = panel.join(oos_merge, how="left").round(4)
    export_df.to_csv(csv_path, index=True)

    fig = plt.figure(figsize=(12, 10), dpi=300)
    gs = gridspec.GridSpec(3, 2, height_ratios=[1.25, 1.05, 1.0], hspace=0.38, wspace=0.25)

    # Panel 1 (Top, full width): Actual Spot USD/BRL vs. Walk-Forward Ridge L2 Forecast
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(
        panel.index,
        panel["usd_brl"],
        color="#1f4e78",
        linewidth=2.0,
        label="Actual Spot USD/BRL (Month-End)",
    )
    ax1.plot(
        oos_df["forecast_date"],
        oos_df["pred_spot_Ridge_L2"],
        color="#d95f02",
        linestyle="--",
        linewidth=1.6,
        alpha=0.9,
        label="Walk-Forward Ridge L2 1M Forecast (OOS)",
    )
    ax1.axvline(
        oos_df["forecast_date"].min(),
        color="#666666",
        linestyle=":",
        linewidth=1.1,
        label="OOS Evaluation Start (Apr 2020)",
    )

    ridge_row = metrics_df.loc[metrics_df["model"] == "Ridge_L2"].iloc[0]
    rw_row = metrics_df.loc[metrics_df["model"] == "Random_Walk"].iloc[0]
    ax1.set_title(
        f"USD/BRL Spot Rate & Expanding-Window Out-of-Sample Ridge Forecast "
        f"(Ridge RMSE: R$ {ridge_row['spot_rmse_brl']:.2f} vs. RW: R$ {rw_row['spot_rmse_brl']:.2f} | "
        f"Dir. Acc.: {ridge_row['dir_accuracy']})",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax1.set_ylabel("BRL per 1 USD", fontweight="bold")
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="upper left", frameon=True, fontsize=9)

    # Panel 2 (Middle, full width): US-Brazil Nominal vs. Real Interest Rate Differential
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(
        panel.index,
        panel["rate_diff"],
        color="#2b8cbe",
        linewidth=1.8,
        label="Nominal Carry Spread (Selic − Fed Funds, pp)",
    )
    ax2.plot(
        panel.index,
        panel["real_rate_diff"],
        color="#1b9e77",
        linewidth=2.0,
        label="Real Carry Spread (Real Selic − Real Fed Funds, pp)",
    )
    ax2.fill_between(
        panel.index,
        0,
        panel["real_rate_diff"],
        where=(panel["real_rate_diff"] < 0),
        color="#d95f02",
        alpha=0.22,
        label="Negative Real Carry Regime",
    )
    ax2.axhline(0, color="#444444", linestyle="--", linewidth=0.9)
    ax2.set_title(
        "Monetary Policy Carry Cushion: Nominal vs. Ex-Post Real Interest Rate Differential (Brazil − US)",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )
    ax2.set_ylabel("Percentage Points (pp)", fontweight="bold")
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="upper right", frameon=True, fontsize=9)

    # Panel 3A (Bottom Left): Standardized Ridge L2 Predictive Feature Weights
    ax3 = fig.add_subplot(gs[2, 0])
    w_plot = weights_df.iloc[::-1]
    label_map = {
        "delta_real_rate_diff_lag1": "Δ Real Carry (t−1)",
        "delta_vix": "Δ VIX Risk Appetite (t)",
        "log_fx_ret": "FX Momentum Return (t)",
        "delta_rate_diff": "Δ Nominal Carry (t)",
        "log_comm_ret_lag1": "Commodity Return (t−1)",
    }
    y_labels = [label_map.get(f, f) for f in w_plot["feature"]]
    bar_colors = ["#1b9e77" if v < 0 else "#d95f02" for v in w_plot["std_ridge_coef"]]

    bars = ax3.barh(y_labels, w_plot["std_ridge_coef"], color=bar_colors, alpha=0.85, height=0.58)
    ax3.axvline(0, color="#444444", linewidth=0.9)
    ax3.set_xlim(-0.85, 0.72)
    ax3.set_title("Standardized Ridge L2 Weights (1M Ahead)", fontsize=10.5, fontweight="bold")
    ax3.set_xlabel("Std. Coefficient (− = BRL Appreciation)", fontsize=9)
    ax3.grid(axis="x", linestyle=":", alpha=0.5)
    for bar in bars:
        val = bar.get_width()
        offset = -0.04 if val < 0 else 0.03
        ha = "right" if val < 0 else "left"
        ax3.text(val + offset, bar.get_y() + bar.get_height() / 2, f"{val:+.2f}", va="center", ha=ha, fontsize=8.5)

    # Panel 3B (Bottom Right): Out-of-Sample Return RMSE vs. Meese-Rogoff Random Walk
    ax4 = fig.add_subplot(gs[2, 1])
    m_plot = metrics_df.sort_values("ret_rmse_pct", ascending=True)
    model_labels = {
        "Random_Walk": "Random Walk (0-Drift)",
        "Random_Forest": "Random Forest",
        "Ridge_L2": "Ridge Regression (L2)",
        "OLS_Linear": "OLS Linear",
        "Gradient_Boosting": "Gradient Boosting",
    }
    x_labels = [model_labels.get(m, m) for m in m_plot["model"]]
    rmse_colors = ["#2b8cbe" if m == "Random_Walk" else "#7570b3" for m in m_plot["model"]]

    bars_rmse = ax4.barh(x_labels, m_plot["ret_rmse_pct"], color=rmse_colors, alpha=0.85, height=0.58)
    ax4.set_xlim(0, 4.65)
    ax4.set_title("OOS Monthly Return RMSE (% — Lower is Better)", fontsize=10.5, fontweight="bold")
    ax4.set_xlabel("Root Mean Squared Error (% MoM)", fontsize=9)
    ax4.grid(axis="x", linestyle=":", alpha=0.5)
    for bar in bars_rmse:
        val = bar.get_width()
        ax4.text(val + 0.06, bar.get_y() + bar.get_height() / 2, f"{val:.2f}%", va="center", ha="left", fontsize=8.5)

    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved dataset to: {csv_path.name} ({len(export_df)} rows, {export_df.shape[1]} columns)")
    print(f"Saved chart to:   {chart_path.name}")