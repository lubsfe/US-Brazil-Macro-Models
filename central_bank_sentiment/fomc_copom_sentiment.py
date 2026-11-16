import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
import matplotlib.pyplot as plt


def fetch_fomc_statement(date_str: str) -> str:
    """Fetches and cleans an FOMC policy statement for a YYYYMMDD date."""
    url = f"https://www.federalreserve.gov/newsevents/pressreleases/monetary{date_str}a.htm"
    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    article_div = soup.find("div", id="article")

    ignore_prefixes = (
        "For release at",
        "Share",
        "Voting for the monetary policy",
        "Last Update:",
    )

    clean_paragraphs = []
    for p in article_div.find_all("p"):
        text = p.get_text().strip()
        if len(text) > 40 and not text.startswith(ignore_prefixes):
            clean_paragraphs.append(text)

    return "\n\n".join(clean_paragraphs)


def fetch_latest_copom_statement() -> tuple[str, str]:
    """Fetches the most recent COPOM statement metadata and cleaned text from BCB."""
    list_url = "https://www.bcb.gov.br/api/servico/sitebcb/copom/comunicados?quantidade=1"
    list_resp = requests.get(list_url).json()

    latest_item = list_resp["conteudo"][0]
    meeting_num = latest_item["nro_reuniao"]
    title = latest_item["titulo"]

    detail_url = f"https://www.bcb.gov.br/api/servico/sitebcb/copom/comunicados_detalhes?nro_reuniao={meeting_num}"
    detail_resp = requests.get(detail_url).json()

    detail_item = detail_resp["conteudo"][0]

    raw_html = detail_item.get("textoComunicado") or detail_item.get("texto", "")
    soup = BeautifulSoup(raw_html, "html.parser")
    clean_text = "\n\n".join(
        p.get_text().strip() for p in soup.find_all("p") if p.get_text().strip()
    )

    return title, clean_text


def score_policy_sentiment(text: str, language: str = "en") -> dict:
    """Computes a normalized Hawkish vs. Dovish score (-1 to +1) for central bank text."""
    lexicons = {
        "en": {
            "hawkish": [
                "inflation", "elevated", "tightening", "restrictive", "upside risks",
                "vigilant", "resilient", "solid pace", "strong", "overheating",
                "pressures", "persistence", "hike", "increase", "anchored"
            ],
            "dovish": [
                "slowed", "moderation", "easing", "accommodative", "downside risks",
                "weakness", "softening", "cut", "lower", "progress",
                "subdued", "disinflation", "support", "unemployment", "gradual"
            ],
        },
        "pt": {
            "hawkish": [
                "inflação", "altistas", "restritiva", "cautela", "aperto",
                "persistência", "pressões", "desancoragem", "elevada", "resiliente",
                "aquecido", "vigilante", "alta", "adverso", "diligência"
            ],
            "dovish": [
                "desaceleração", "moderação", "flexibilização", "queda", "corte",
                "redução", "baixistas", "desinflação", "arrefecimento", "ociosidade",
                "convergência", "alívio", "gradual", "estabilidade", "benigno"
            ],
        },
    }

    text_lower = text.lower()
    hawk_terms = lexicons[language]["hawkish"]
    dove_terms = lexicons[language]["dovish"]

    hawk_count = sum(
        len(re.findall(rf"\b{re.escape(term)}\b", text_lower)) for term in hawk_terms
    )
    dove_count = sum(
        len(re.findall(rf"\b{re.escape(term)}\b", text_lower)) for term in dove_terms
    )

    total = hawk_count + dove_count
    net_score = (hawk_count - dove_count) / total if total > 0 else 0.0

    return {
        "hawkish_hits": hawk_count,
        "dovish_hits": dove_count,
        "net_score": round(net_score, 3),
    }


def build_fomc_sentiment_history(limit: int = 20) -> pd.DataFrame:
    """Scrapes the most recent FOMC statement dates and scores their policy tone."""
    calendar_url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    resp = requests.get(calendar_url)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    date_pattern = re.compile(r"/newsevents/pressreleases/monetary(\d{8})a\.htm")

    dates = sorted(
        {
            match.group(1)
            for a in soup.find_all("a", href=True)
            if (match := date_pattern.search(a["href"]))
        },
        reverse=True,
    )[:limit]

    records = []
    for date_str in dates:
        text = fetch_fomc_statement(date_str)
        scores = score_policy_sentiment(text, language="en")
        records.append({
            "date": pd.to_datetime(date_str, format="%Y%m%d"),
            "central_bank": "FOMC",
            **scores,
        })
        time.sleep(0.2)  # Pause briefly between requests

    return pd.DataFrame(records).sort_values("date").reset_index(drop=True)


def build_copom_sentiment_history(limit: int = 20) -> pd.DataFrame:
    """Fetches the most recent COPOM statements from BCB and scores their policy tone."""
    list_url = f"https://www.bcb.gov.br/api/servico/sitebcb/copom/comunicados?quantidade={limit}"
    list_resp = requests.get(list_url).json()

    records = []
    for item in list_resp["conteudo"]:
        meeting_num = item["nro_reuniao"]
        meeting_date = pd.to_datetime(item["dataReferencia"][:10])

        detail_url = f"https://www.bcb.gov.br/api/servico/sitebcb/copom/comunicados_detalhes?nro_reuniao={meeting_num}"
        detail_resp = requests.get(detail_url).json()
        raw_html = detail_resp["conteudo"][0].get("textoComunicado", "")

        # Extract all text safely regardless of whether paragraphs use <p> or <div> tags
        clean_text = BeautifulSoup(raw_html, "html.parser").get_text(separator=" ").strip()
        scores = score_policy_sentiment(clean_text, language="pt")

        records.append({
            "date": meeting_date,
            "central_bank": "COPOM",
            **scores,
        })
        time.sleep(0.2)

    return pd.DataFrame(records).sort_values("date").reset_index(drop=True)


def fetch_policy_rates(start_date: pd.Timestamp, end_date: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetches daily US Fed Funds upper target (FRED) and Brazilian Selic target (BCB SGS 432)."""
    # 1. Fetch US Federal Funds Target Range Upper Limit (DFEDTARU) from FRED
    fred_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFEDTARU"
    df_fed = pd.read_csv(fred_url)
    df_fed.columns = ["date", "policy_rate"]
    df_fed["date"] = pd.to_datetime(df_fed["date"])
    df_fed["policy_rate"] = pd.to_numeric(df_fed["policy_rate"], errors="coerce")
    df_fed = df_fed.dropna().sort_values("date").reset_index(drop=True)

    # 2. Fetch Brazilian Selic Target Rate (Series 432) from BCB SGS API
    dt_ini = (start_date - pd.Timedelta(days=10)).strftime("%d/%m/%Y")
    dt_fim = (end_date + pd.Timedelta(days=10)).strftime("%d/%m/%Y")
    bcb_url = (
        f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados"
        f"?formato=json&dataInicial={dt_ini}&dataFinal={dt_fim}"
    )
    bcb_json = requests.get(bcb_url, headers={"Accept": "application/json"}, timeout=10).json()
    df_selic = pd.DataFrame(bcb_json)
    df_selic["date"] = pd.to_datetime(df_selic["data"], format="%d/%m/%Y")
    df_selic["policy_rate"] = pd.to_numeric(df_selic["valor"], errors="coerce")
    df_selic = df_selic[["date", "policy_rate"]].dropna().sort_values("date").reset_index(drop=True)

    return df_fed, df_selic


def attach_rate_decisions(df_sent: pd.DataFrame, df_rates: pd.DataFrame) -> pd.DataFrame:
    """Attaches pre-meeting rate, post-meeting rate, and decision size (bps) to each meeting."""
    df_m = df_sent.sort_values("date").copy()
    df_m["pre_lookup"] = df_m["date"] - pd.Timedelta(days=1)
    df_m["post_lookup"] = df_m["date"] + pd.Timedelta(days=3)

    pre = pd.merge_asof(
        df_m[["date", "pre_lookup"]],
        df_rates,
        left_on="pre_lookup",
        right_on="date",
        direction="backward",
    )["policy_rate"]

    post = pd.merge_asof(
        df_m[["date", "post_lookup"]],
        df_rates,
        left_on="post_lookup",
        right_on="date",
        direction="backward",
    )["policy_rate"]

    df_m["pre_rate"] = pre.values
    df_m["post_rate"] = post.values
    df_m["rate_change_bps"] = ((df_m["post_rate"] - df_m["pre_rate"]) * 100).round(0)
    return df_m.drop(columns=["pre_lookup", "post_lookup"])


def export_results_and_chart(df_all: pd.DataFrame) -> None:
    """Saves the merged sentiment-rate dataset to CSV and renders a two-panel comparison chart."""
    output_dir = Path(__file__).resolve().parent
    csv_path = output_dir / "fomc_copom_sentiment_data.csv"
    chart_path = output_dir / "fomc_copom_sentiment_chart.png"

    df_all.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)
    configs = [
        ("FOMC", "US Federal Reserve (FOMC): Statement Tone vs. Fed Funds Target", "#1f77b4", "#d62728"),
        ("COPOM", "Banco Central do Brasil (COPOM): Statement Tone vs. Selic Target", "#2ca02c", "#ff7f0e"),
    ]

    for ax1, (bank, title, bar_color, line_color) in zip(axes, configs):
        sub = df_all[df_all["central_bank"] == bank].sort_values("date")

        ax1.bar(sub["date"], sub["net_score"], width=18, color=bar_color, alpha=0.45, label="Net Hawkishness (-1 to +1)")
        ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax1.set_ylabel("Net Hawkishness Score", color=bar_color, fontweight="bold")
        ax1.set_ylim(-1.0, 1.0)
        ax1.set_title(title, fontsize=11, fontweight="bold")
        ax1.grid(axis="y", linestyle=":", alpha=0.5)

        ax2 = ax1.twinx()
        ax2.plot(sub["date"], sub["post_rate"], color=line_color, marker="o", linewidth=2.2, label="Post-Meeting Policy Rate (%)")
        ax2.set_ylabel("Policy Rate (%)", color=line_color, fontweight="bold")

    plt.tight_layout()
    plt.savefig(chart_path, dpi=300)
    plt.close(fig)
    print(f"\nSaved dataset to: {csv_path.name}")
    print(f"Saved chart to:   {chart_path.name}")


# --- STEP 5 TEST RUN ---
print("Fetching and scoring historical FOMC statements...")
df_fomc_sent = build_fomc_sentiment_history(limit=20)

print("Fetching and scoring historical COPOM statements...")
df_copom_sent = build_copom_sentiment_history(limit=20)

min_dt = min(df_fomc_sent["date"].min(), df_copom_sent["date"].min())
max_dt = max(df_fomc_sent["date"].max(), df_copom_sent["date"].max())

print("Fetching FRED and BCB policy interest rates...")
df_fed_rates, df_selic_rates = fetch_policy_rates(min_dt, max_dt)

df_fomc_full = attach_rate_decisions(df_fomc_sent, df_fed_rates)
df_copom_full = attach_rate_decisions(df_copom_sent, df_selic_rates)
df_final = pd.concat([df_fomc_full, df_copom_full], ignore_index=True)

print("\n--- MERGED SENTIMENT & RATE DECISIONS (RECENT MEETINGS) ---")
print(df_final.groupby("central_bank").tail(5).to_string(index=False))

print("\n--- CORRELATION: NET HAWKISHNESS VS. RATE CHANGE (BPS) ---")
for bank, grp in df_final.groupby("central_bank"):
    corr = grp["net_score"].corr(grp["rate_change_bps"])
    print(f"{bank}: r = {corr:.3f}")

export_results_and_chart(df_final)