import io
import time
from datetime import datetime

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import DEFAULT_START_DATE


def _build_retrying_session() -> requests.Session:
    """Creates an HTTP session with automatic retries for BCB gateway hiccups."""
    retry_strategy = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    })
    return session


def fetch_fred_series(
        series_id: str,
        value_col: str,
        start_date: str = DEFAULT_START_DATE,
) -> pd.DataFrame:
    """Fetches a time series directly from FRED's CSV endpoint using pandas (bypassing requests pooling)."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url)
    df.columns = ["date", value_col]
    df["date"] = pd.to_datetime(df["date"])
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    df = (
        df.dropna(subset=[value_col])
        .loc[lambda d: d["date"] >= pd.to_datetime(start_date)]
        .sort_values("date")
        .reset_index(drop=True)
    )
    return df


def fetch_bcb_sgs_series(
        series_id: int,
        value_col: str,
        start_date: str = DEFAULT_START_DATE,
        end_date: str | None = None,
        session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetches a BCB SGS series in <=8-year chunks with automatic retries for BCB gateway hiccups."""
    http = session or _build_retrying_session()
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date) if end_date else pd.Timestamp(datetime.today().date())

    records: list[dict] = []
    chunk_start = start_dt

    while chunk_start <= end_dt:
        chunk_end = min(chunk_start + pd.DateOffset(years=8) - pd.Timedelta(days=1), end_dt)
        url = (
            f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_id}/dados"
            f"?formato=json"
            f"&dataInicial={chunk_start.strftime('%d/%m/%Y')}"
            f"&dataFinal={chunk_end.strftime('%d/%m/%Y')}"
        )
        resp = http.get(url, timeout=25)
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, list):
            records.extend(payload)

        chunk_start = chunk_end + pd.Timedelta(days=1)
        time.sleep(0.4)

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df[value_col] = pd.to_numeric(df["valor"], errors="coerce")

    df = (
        df[["date", value_col]]
        .dropna(subset=[value_col])
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    return df


def fetch_all_macro_series(start_date: str = DEFAULT_START_DATE) -> dict[str, pd.DataFrame]:
    """Pulls all raw USD/BRL, rate, inflation, volatility, and commodity series."""
    bcb_session = _build_retrying_session()
    series_registry = {
        "usd_brl": ("FRED:DEXBZUS", lambda: fetch_fred_series("DEXBZUS", "usd_brl", start_date)),
        "fed_funds": ("FRED:DFF", lambda: fetch_fred_series("DFF", "fed_funds", start_date)),
        "selic_target": ("BCB:432", lambda: fetch_bcb_sgs_series(432, "selic_target", start_date, session=bcb_session)),
        "us_cpi": ("FRED:CPIAUCSL", lambda: fetch_fred_series("CPIAUCSL", "us_cpi", start_date)),
        "br_ipca_mom": ("BCB:433", lambda: fetch_bcb_sgs_series(433, "br_ipca_mom", start_date, session=bcb_session)),
        "vix": ("FRED:VIXCLS", lambda: fetch_fred_series("VIXCLS", "vix", start_date)),
        "commodities": ("FRED:PALLFNFINDEXM", lambda: fetch_fred_series("PALLFNFINDEXM", "commodities", start_date)),
    }

    raw_data: dict[str, pd.DataFrame] = {}
    for key, (source_label, loader) in series_registry.items():
        print(f"Fetching {key:<14} ({source_label})...")
        raw_data[key] = loader()
        time.sleep(0.25)

    return raw_data


def verify_raw_ingestion(raw_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Builds a diagnostic summary table confirming row counts, date bounds, and latest values."""
    rows = []
    for key, df in raw_data.items():
        latest_row = df.iloc[-1]
        rows.append({
            "series": key,
            "rows": len(df),
            "start_date": df["date"].min().strftime("%Y-%m-%d"),
            "end_date": df["date"].max().strftime("%Y-%m-%d"),
            "latest_value": round(float(latest_row[key]), 4),
        })
    return pd.DataFrame(rows)