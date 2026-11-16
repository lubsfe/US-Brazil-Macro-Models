from config import DEFAULT_START_DATE, PANEL_START_DATE, PROJECT_DIR
from data_ingestion import fetch_all_macro_series
from feature_engineering import build_monthly_macro_panel, run_adf_stationarity_tests
from models import run_econometric_baselines, run_walk_forward_validation
from visualization import export_artifacts


def main() -> None:
    """Runs the end-to-end USD/BRL macro ingestion, econometric, ML backtest, and export pipeline."""
    print("=== STEP 1: MULTI-SOURCE MACRO API INGESTION ===")
    raw_series = fetch_all_macro_series(start_date=DEFAULT_START_DATE)

    print("\n=== STEP 2: MONTHLY PANEL ALIGNMENT & STATIONARITY CHECKS ===")
    macro_panel = build_monthly_macro_panel(raw_series, panel_start=PANEL_START_DATE)
    print(
        f"Aligned monthly panel: {len(macro_panel)} months "
        f"({macro_panel.index.min().strftime('%Y-%m-%d')} to {macro_panel.index.max().strftime('%Y-%m-%d')})"
    )

    adf_df = run_adf_stationarity_tests(macro_panel)
    print("\n--- AUGMENTED DICKEY-FULLER (ADF) STATIONARITY TESTS ---")
    print(adf_df.to_string(index=False))

    print("\n=== STEP 3: NEWEY-WEST HAC ECONOMETRIC BASELINES ===")
    econ_results = run_econometric_baselines(macro_panel)
    m_contemp = econ_results["contemp_model"]
    m_pred = econ_results["pred_model"]
    print(
        f"Model A (Contemporaneous): R2 = {m_contemp.rsquared:.4f}, F = {m_contemp.fvalue:.3f} (p = {m_contemp.f_pvalue:.4f})\n"
        f"Model B (1-Mo Predictive): R2 = {m_pred.rsquared:.4f}, F = {m_pred.fvalue:.3f} (p = {m_pred.f_pvalue:.4f})"
    )

    print("\n=== STEP 4: EXPANDING-WINDOW WALK-FORWARD ML VALIDATION ===")
    wf_results = run_walk_forward_validation(macro_panel, min_train_months=60)
    print(wf_results["metrics"].to_string(index=False))

    print("\n=== STEP 5: EXPORTING DATASET & DIAGNOSTIC VISUALIZATION ===")
    export_artifacts(macro_panel, wf_results, output_dir=PROJECT_DIR)


if __name__ == "__main__":
    main()