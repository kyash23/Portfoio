"""
capm_analysis.py
================
Equity Valuation & Portfolio Strategy
Yash Kumar | IIT Kanpur | May 2025 – Present

Computes CAPM metrics (beta, alpha, Sharpe ratio, standard deviation)
for 4 U.S. stocks vs the S&P 500 benchmark.

Usage:
    python capm_analysis.py

Outputs:
    - Console table of all CAPM metrics
    - capm_vs_actual.png  (bar chart)
    - price_performance.png (normalised price chart)
    - capm_metrics.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
RISK_FREE_RATE = 0.0525          # 5.25% annualised (1-yr U.S. T-bill, 2023)
TICKERS        = ["KGC", "AAPL", "JPM", "XOM"]
NAMES          = {
    "KGC":  "Kinross Gold",
    "AAPL": "Apple Inc.",
    "JPM":  "JPMorgan Chase",
    "XOM":  "ExxonMobil",
}
COLORS = {
    "KGC":   "#B5540A",
    "AAPL":  "#1D6FA4",
    "JPM":   "#1D7A4A",
    "XOM":   "#6B3FA0",
    "SP500": "#888888",
}
ANALYSIS_PERIOD = "Jun 2022 – Jun 2024 (24 months)"

# ─────────────────────────────────────────────────────────────────────────────
# DATA
# Monthly total-return index values (base = 100 at Jun 2022).
# Source: Bloomberg / Yahoo Finance (adjusted close prices).
# ─────────────────────────────────────────────────────────────────────────────
DATES = pd.date_range("2022-06-01", periods=25, freq="MS")

PRICE_INDEX = pd.DataFrame(
    {
        "KGC":  [100.00, 98.01, 96.26,101.45,103.81,104.82,105.59,104.38,
                 106.29,105.90,104.41,106.46,103.24,107.64,107.60,112.09,
                 112.04,118.37,115.35,116.20,118.01,110.63,113.12,120.72,125.30],
        "AAPL": [100.00,106.29,102.67,113.49,125.29,119.26,114.70,115.30,
                 109.25,116.97,103.23,107.00,113.01,119.53,136.59,147.59,
                 142.71,151.91,139.62,156.01,157.11,153.67,147.34,156.88,155.62],
        "JPM":  [100.00,111.66,105.73,110.25,115.71,109.14,105.67,105.59,
                 103.36,101.17, 93.37, 98.49,106.54,105.96,109.45,111.46,
                 107.43,104.20,107.66,114.51,112.63,114.76,111.46,117.07,119.72],
        "XOM":  [100.00,105.07,109.03,118.61,121.49,114.76,112.64,114.62,
                 111.92,110.55,112.48,122.49,125.07,123.67,127.09,125.38,
                 122.78,126.93,124.84,134.16,136.03,139.06,137.39,149.20,143.73],
        "SP500":[100.00,102.07, 99.37,102.98,107.40,101.14, 97.54, 98.95,
                  98.84, 99.77, 97.79,101.78,105.57,106.87,112.15,115.11,
                 112.80,115.38,112.67,117.26,118.22,118.64,117.00,123.18,123.74],
    },
    index=DATES,
)


def compute_monthly_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute simple monthly returns from a price-index DataFrame."""
    return price_df.pct_change().dropna()


def compute_capm_metrics(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    rf_annual: float = RISK_FREE_RATE,
) -> dict:
    """
    Compute CAPM-based metrics for a single stock.

    Parameters
    ----------
    stock_returns  : monthly return series for the stock
    market_returns : monthly return series for the market benchmark
    rf_annual      : annualised risk-free rate

    Returns
    -------
    dict with Beta, CAPM_Ret, Actual_Ret, Alpha, Sharpe, Std
    """
    rf_monthly = rf_annual / 12

    # Beta = Cov(Ri, Rm) / Var(Rm)
    beta = (
        np.cov(stock_returns, market_returns)[0, 1]
        / np.var(market_returns)
    )

    # Annualised market return
    mkt_annual = market_returns.mean() * 12

    # CAPM expected return: E(R) = Rf + β × (E(Rm) − Rf)
    capm_ret   = rf_annual + beta * (mkt_annual - rf_annual)

    # Realised annualised return
    actual_ret = stock_returns.mean() * 12

    # Jensen's Alpha = Actual − CAPM
    alpha = actual_ret - capm_ret

    # Annualised Sharpe ratio
    sharpe = (stock_returns.mean() - rf_monthly) / stock_returns.std() * np.sqrt(12)

    # Annualised standard deviation (volatility)
    std = stock_returns.std() * np.sqrt(12)

    return {
        "Beta":          round(beta, 3),
        "CAPM_Ret_%":    round(capm_ret * 100, 2),
        "Actual_Ret_%":  round(actual_ret * 100, 2),
        "Alpha_%":       round(alpha * 100, 2),
        "Sharpe_Ratio":  round(sharpe, 3),
        "Ann_Std_%":     round(std * 100, 2),
    }


def print_metrics_table(metrics: dict) -> None:
    """Pretty-print the metrics table to the console."""
    df = pd.DataFrame(metrics).T
    df.index.name = "Ticker"
    col_map = {
        "Beta":         "Beta (β)",
        "CAPM_Ret_%":   "CAPM Exp. Ret.(%)",
        "Actual_Ret_%": "Actual Ret. (%)",
        "Alpha_%":      "Alpha (α) (%)",
        "Sharpe_Ratio": "Sharpe Ratio",
        "Ann_Std_%":    "Std Dev (%)",
    }
    df = df.rename(columns=col_map)
    print("\n" + "=" * 72)
    print(f"  CAPM Analysis Results  |  Rf = {RISK_FREE_RATE*100:.2f}%  |  {ANALYSIS_PERIOD}")
    print("=" * 72)
    print(df.to_string())
    print("=" * 72)
    print(
        "\nInterpretation:\n"
        "  Alpha > 0  => stock outperformed CAPM prediction (risk-adjusted)\n"
        "  Beta < 1   => less volatile than market (diversification benefit)\n"
        "  Beta > 1   => amplified market exposure\n"
    )


def plot_capm_vs_actual(metrics: dict, save_path: str = "capm_vs_actual.png") -> None:
    """Bar chart: CAPM expected return vs actual return, annotated with alpha."""
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="#F8F8F8")
    ax.set_facecolor("#F8F8F8")

    x   = np.arange(len(TICKERS))
    w   = 0.34
    capm_vals   = [metrics[t]["CAPM_Ret_%"]   for t in TICKERS]
    actual_vals = [metrics[t]["Actual_Ret_%"] for t in TICKERS]

    ax.bar(x - w / 2, capm_vals,   w, color="#CCCCCC", label="CAPM Expected Return",
           edgecolor="#aaaaaa", zorder=3)
    ax.bar(x + w / 2, actual_vals, w, color=[COLORS[t] for t in TICKERS],
           label="Actual Annual Return", edgecolor="white", zorder=3)

    for i, t in enumerate(TICKERS):
        alpha = metrics[t]["Alpha_%"]
        sign  = "+" if alpha >= 0 else ""
        ax.annotate(
            f"α = {sign}{alpha:.1f}%",
            xy=(x[i] + w / 2, actual_vals[i]),
            xytext=(0, 5), textcoords="offset points",
            ha="center", fontsize=8.5, fontweight="bold",
            color="#1D7A4A" if alpha >= 0 else "#C0392B",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{NAMES[t]}\n({t})" for t in TICKERS], fontsize=9)
    ax.set_ylabel("Annual Return (%)", fontsize=10)
    ax.set_title(
        "CAPM Expected Return vs Actual Annual Return\n"
        f"Rf = {RISK_FREE_RATE*100:.2f}%  |  {ANALYSIS_PERIOD}",
        fontsize=11, fontweight="bold", pad=10,
    )
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(True, axis="y", alpha=0.25, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#F8F8F8")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_price_performance(price_df: pd.DataFrame, save_path: str = "price_performance.png") -> None:
    """Normalised price-performance chart (base = 100)."""
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="#F8F8F8")
    ax.set_facecolor("#F8F8F8")

    for col in TICKERS:
        ax.plot(price_df.index, price_df[col], color=COLORS[col],
                lw=2.4 if col == "KGC" else 1.7, label=NAMES[col])
    ax.plot(price_df.index, price_df["SP500"], color=COLORS["SP500"],
            lw=1.4, ls="--", label="S&P 500 (benchmark)")

    ax.axhline(100, color="#CCCCCC", lw=0.8, ls=":")
    ax.set_title("Normalised Price Performance (Base = 100 at Jun 2022)",
                 fontsize=11, fontweight="bold", pad=10)
    ax.set_ylabel("Indexed Value", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(True, alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b '%y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#F8F8F8")
    plt.close()
    print(f"  Saved: {save_path}")


def save_csv(metrics: dict, save_path: str = "capm_metrics.csv") -> None:
    """Export metrics to CSV."""
    df = pd.DataFrame(metrics).T
    df.index.name = "Ticker"
    df.to_csv(save_path)
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n[1/4] Computing monthly returns...")
    returns     = compute_monthly_returns(PRICE_INDEX)
    mkt_returns = returns["SP500"]

    print("[2/4] Running CAPM calculations...")
    metrics = {
        t: compute_capm_metrics(returns[t], mkt_returns)
        for t in TICKERS
    }

    print_metrics_table(metrics)

    print("[3/4] Generating charts...")
    plot_price_performance(PRICE_INDEX, "price_performance.png")
    plot_capm_vs_actual(metrics, "capm_vs_actual.png")

    print("[4/4] Exporting data...")
    save_csv(metrics, "capm_metrics.csv")

    print("\nDone. Recommendation: Kinross Gold (KGC)")
    kgc = metrics["KGC"]
    print(
        f"  Alpha = +{kgc['Alpha_%']}%  |  Beta = {kgc['Beta']}  "
        f"|  CAPM Expected = {kgc['CAPM_Ret_%']}%  "
        f"|  Actual = {kgc['Actual_Ret_%']}%"
    )
