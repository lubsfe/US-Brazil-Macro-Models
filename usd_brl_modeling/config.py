from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_START_DATE = "2014-01-01"
PANEL_START_DATE = "2015-01-31"

PREDICTIVE_FEATURES = [
    "delta_real_rate_diff_lag1",
    "delta_rate_diff",
    "log_comm_ret_lag1",
    "delta_vix",
    "log_fx_ret",
]