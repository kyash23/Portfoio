"""
excel_metrics.py
================
Equity Valuation & Portfolio Strategy
Yash Kumar | IIT Kanpur | May 2025 – Present

Replicates the Excel-based metric computations (alpha, beta, Sharpe ratio,
standard deviation) showing step-by-step workings for each stock — exactly
as done in Excel for the project.

Usage:
    python excel_metrics.py

Outputs:
    - Verbose console walkthrough of every calculation
    - excel_metrics_detailed.csv
    - beta_regression.png  (scatter + regression line for each stock vs S&P 500)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG & DATA  (same source as capm_analysis.py)
# ─────────────────────────────────────────────────────────────────────────────
RISK_FREE_RATE = 0.0525   # 5.25% annualised
TICKERS        = ["KGC", "AAPL", "JPM", "XOM"]
NAMES          = {
    "KGC":  "Kinross Gold",
    "AAPL": "Apple Inc.",
    "JPM":  "JPMorgan Chase",
    "XOM":  "ExxonMobil",
}
COLORS = {"KGC": "#B5540A", "AAPL": "#1D6FA4", "JPM": "#1D7A4A", "XOM": "#6B3FA0"}

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


# ─────────────────────────────────────────────────────────────────────────────
# STEP-BY-STEP CALCULATIONS  (mirrors Excel formula logic)
# ─────────────────────────────────────────────────────────────────────────────
def excel_walkthrough(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute metrics step by step, printing Excel-equivalent formula logic.

    Excel formulas used:
        Beta    = COVAR(Ri, Rm) / VAR(Rm)           [=SLOPE(Ri_range, Rm_range)]
        Alpha   = Actual_Return - CAPM_Return
        Sharpe  = (Mean_Monthly_Ret - Rf_Monthly) / Stdev_Monthly * SQRT(12)
        Std_Dev = STDEV(monthly_returns) * SQRT(12)
    """
    mkt    = returns["SP500"]
    rf_m   = RISK_FREE_RATE / 12
    E_mkt  = mkt.mean() * 12

    records = []
    sep = "-" * 64

    for t in TICKERS:
        r = returns[t]
        print(f"\n{'='*64}")
        print(f"  {NAMES[t]} ({t})")
        print(sep)

        # ── Beta ─────────────────────────────────────────────────────────────
        cov_ri_rm = np.cov(r, mkt, ddof=1)[0, 1]
        var_rm    = np.var(mkt, ddof=1)
        beta      = cov_ri_rm / var_rm
        print(f"  Excel: =SLOPE(monthly_returns, sp500_returns)")
        print(f"       = COV(Ri,Rm) / VAR(Rm)")
        print(f"       = {cov_ri_rm:.6f} / {var_rm:.6f}")
        print(f"  Beta  = {beta:.4f}")

        # ── CAPM Expected Return ──────────────────────────────────────────────
        capm_ret = RISK_FREE_RATE + beta * (E_mkt - RISK_FREE_RATE)
        print(f"\n  Excel: =Rf + Beta*(E_Rm - Rf)")
        print(f"       = {RISK_FREE_RATE:.4f} + {beta:.4f}*({E_mkt:.4f} - {RISK_FREE_RATE:.4f})")
        print(f"  CAPM Expected Return = {capm_ret*100:.2f}%")

        # ── Actual Return ─────────────────────────────────────────────────────
        actual_ret = r.mean() * 12
        print(f"\n  Excel: =AVERAGE(monthly_returns)*12")
        print(f"       = {r.mean():.6f} * 12")
        print(f"  Actual Annual Return = {actual_ret*100:.2f}%")

        # ── Alpha ─────────────────────────────────────────────────────────────
        alpha = actual_ret - capm_ret
        print(f"\n  Excel: =Actual_Return - CAPM_Return")
        print(f"       = {actual_ret*100:.2f}% - {capm_ret*100:.2f}%")
        print(f"  Alpha = {alpha*100:+.2f}%", "  ← Outperforms CAPM" if alpha > 0 else "")

        # ── Standard Deviation ────────────────────────────────────────────────
        monthly_std = r.std(ddof=1)
        ann_std     = monthly_std * np.sqrt(12)
        print(f"\n  Excel: =STDEV(monthly_returns)*SQRT(12)")
        print(f"       = {monthly_std:.6f} * {np.sqrt(12):.4f}")
        print(f"  Annual Std Dev = {ann_std*100:.2f}%")

        # ── Sharpe Ratio ──────────────────────────────────────────────────────
        sharpe = (r.mean() - rf_m) / monthly_std * np.sqrt(12)
        print(f"\n  Excel: =(AVERAGE(Ri)-Rf_monthly)/STDEV(Ri)*SQRT(12)")
        print(f"       = ({r.mean():.6f} - {rf_m:.6f}) / {monthly_std:.6f} * {np.sqrt(12):.4f}")
        print(f"  Sharpe Ratio = {sharpe:.3f}")

        records.append({
            "Ticker":        t,
            "Name":          NAMES[t],
            "Beta":          round(beta, 3),
            "CAPM_Ret_%":    round(capm_ret * 100, 2),
            "Actual_Ret_%":  round(actual_ret * 100, 2),
            "Alpha_%":       round(alpha * 100, 2),
            "Ann_Std_%":     round(ann_std * 100, 2),
            "Sharpe_Ratio":  round(sharpe, 3),
        })

    return pd.DataFrame(records).set_index("Ticker")


def plot_beta_regressions(returns: pd.DataFrame,
                          save_path: str = "beta_regression.png") -> None:
    """
    2×2 scatter plots: each stock's monthly returns vs S&P 500,
    with OLS regression line (= beta slope in Excel's SLOPE function).
    """
    mkt = returns["SP500"]
    fig = plt.figure(figsize=(11, 8), facecolor="#F8F8F8")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    for idx, t in enumerate(TICKERS):
        ax  = fig.add_subplot(gs[idx // 2, idx % 2])
        ax.set_facecolor("#F8F8F8")
        r   = returns[t]
        col = COLORS[t]

        # Scatter
        ax.scatter(mkt * 100, r * 100, color=col, alpha=0.65, s=30, zorder=3)

        # Regression line  (same as Excel SLOPE / INTERCEPT)
        coeffs    = np.polyfit(mkt, r, 1)
        beta_val  = coeffs[0]
        intercept = coeffs[1]
        x_line    = np.linspace(mkt.min(), mkt.max(), 100)
        ax.plot(x_line * 100, (beta_val * x_line + intercept) * 100,
                color=col, lw=2, zorder=4,
                label=f"β = {beta_val:.3f}  |  α = {intercept*12*100:+.2f}%/yr")

        ax.axhline(0, color="#CCCCCC", lw=0.7, ls="--")
        ax.axvline(0, color="#CCCCCC", lw=0.7, ls="--")
        ax.set_xlabel("S&P 500 Monthly Return (%)", fontsize=8.5)
        ax.set_ylabel(f"{t} Monthly Return (%)", fontsize=8.5)
        ax.set_title(f"{NAMES[t]} ({t})", fontsize=9.5, fontweight="bold", color=col)
        ax.legend(fontsize=8, framealpha=0.85)
        ax.grid(True, alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Beta Estimation: Monthly Returns vs S&P 500\n"
        "(regression slope = β; matches Excel SLOPE function)",
        fontsize=11, fontweight="bold", y=1.01,
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#F8F8F8")
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 64)
    print("  EXCEL METRIC WALKTHROUGH — Equity Valuation & Portfolio Strategy")
    print("  Rf = 5.25%  |  Jun 2022 – Jun 2024  |  24 monthly observations")
    print("=" * 64)

    returns = PRICE_INDEX.pct_change().dropna()
    df      = excel_walkthrough(returns)

    print("\n\n" + "=" * 64)
    print("  SUMMARY TABLE")
    print("=" * 64)
    print(df.to_string())

    print("\n\nGenerating beta regression chart...")
    plot_beta_regressions(returns)

    df.to_csv("excel_metrics_detailed.csv")
    print("  Saved: excel_metrics_detailed.csv")
    print("\nDone.")
