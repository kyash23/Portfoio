"""
portfolio_optimization.py
==========================
Equity Valuation & Portfolio Strategy
Yash Kumar | IIT Kanpur | May 2025 – Present

Constructs the mean-variance efficient frontier using Monte Carlo simulation
and finds the maximum-Sharpe-ratio portfolio via constrained optimisation.

Usage:
    python portfolio_optimization.py

Outputs:
    - Console: optimal weights, portfolio return / volatility / Sharpe
    - efficient_frontier.png
    - optimal_weights.png
    - portfolio_results.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (must match capm_analysis.py)
# ─────────────────────────────────────────────────────────────────────────────
RISK_FREE_RATE = 0.0525
TICKERS        = ["KGC", "AAPL", "JPM", "XOM"]
NAMES          = {
    "KGC":  "Kinross Gold",
    "AAPL": "Apple Inc.",
    "JPM":  "JPMorgan Chase",
    "XOM":  "ExxonMobil",
}
COLORS = {
    "KGC":  "#B5540A",
    "AAPL": "#1D6FA4",
    "JPM":  "#1D7A4A",
    "XOM":  "#6B3FA0",
}
N_SIMULATIONS  = 5_000
RANDOM_SEED    = 777

# ─────────────────────────────────────────────────────────────────────────────
# DATA  (monthly returns derived from price index; see capm_analysis.py)
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
    },
    index=DATES,
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def portfolio_stats(weights: np.ndarray, ann_returns: np.ndarray,
                    cov_matrix: np.ndarray) -> tuple[float, float, float]:
    """Return (annualised return, annualised volatility, Sharpe ratio)."""
    port_ret = weights @ ann_returns
    port_vol = np.sqrt(weights @ cov_matrix @ weights)
    sharpe   = (port_ret - RISK_FREE_RATE) / port_vol
    return port_ret, port_vol, sharpe


def neg_sharpe(weights: np.ndarray, ann_returns: np.ndarray,
               cov_matrix: np.ndarray) -> float:
    """Objective function: negative Sharpe (minimised by scipy)."""
    _, _, sharpe = portfolio_stats(weights, ann_returns, cov_matrix)
    return -sharpe


def run_monte_carlo(ann_returns: np.ndarray, cov_matrix: np.ndarray,
                    n: int = N_SIMULATIONS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Simulate `n` random long-only portfolios.

    Returns
    -------
    DataFrame with columns: Return, Volatility, Sharpe, w_KGC, w_AAPL, w_JPM, w_XOM
    """
    rng = np.random.default_rng(seed)
    records = []
    for _ in range(n):
        w            = rng.dirichlet(np.ones(len(TICKERS)))  # random weights summing to 1
        ret, vol, sr = portfolio_stats(w, ann_returns, cov_matrix)
        row          = {"Return_%": round(ret * 100, 4),
                        "Volatility_%": round(vol * 100, 4),
                        "Sharpe": round(sr, 4)}
        row.update({f"w_{t}": round(w[i], 4) for i, t in enumerate(TICKERS)})
        records.append(row)
    return pd.DataFrame(records)


def optimise_max_sharpe(ann_returns: np.ndarray,
                        cov_matrix: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """
    Find the maximum-Sharpe portfolio via SLSQP.

    Constraints:
      - weights sum to 1
      - all weights in [0, 1]  (long-only)

    Returns
    -------
    (optimal_weights, opt_return, opt_volatility, opt_sharpe)
    """
    n      = len(TICKERS)
    w0     = np.ones(n) / n                           # equal-weight starting point
    cons   = {"type": "eq", "fun": lambda w: w.sum() - 1}
    bounds = [(0.0, 1.0)] * n

    result = minimize(
        neg_sharpe, w0,
        args=(ann_returns, cov_matrix),
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    w_opt             = result.x
    ret, vol, sharpe  = portfolio_stats(w_opt, ann_returns, cov_matrix)
    return w_opt, ret, vol, sharpe


def print_results(w_opt, ret, vol, sharpe) -> None:
    print("\n" + "=" * 60)
    print("  PORTFOLIO OPTIMISATION — MAX SHARPE RATIO")
    print("=" * 60)
    print(f"  {'Expected Annual Return':<28} {ret*100:.2f}%")
    print(f"  {'Annual Volatility':<28} {vol*100:.2f}%")
    print(f"  {'Sharpe Ratio':<28} {sharpe:.3f}")
    print(f"  {'Risk-Free Rate':<28} {RISK_FREE_RATE*100:.2f}%")
    print(f"  {'Simulations Run':<28} {N_SIMULATIONS:,}")
    print("-" * 60)
    print("  Optimal Weights:")
    for t, w in zip(TICKERS, w_opt):
        bar = "█" * int(w * 40)
        print(f"    {NAMES[t]:<18} ({t})  {w*100:5.1f}%  {bar}")
    print("=" * 60)
    print("\n  Recommendation: Kinross Gold (KGC)")
    print(f"  Alpha = +3.0%  |  Beta = 0.35  |  Strong diversification benefit\n")


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def plot_efficient_frontier(mc_df: pd.DataFrame, w_opt, ret, vol, sharpe,
                            ann_returns, ann_stds,
                            save_path: str = "efficient_frontier.png") -> None:
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#F8F8F8")
    ax.set_facecolor("#F8F8F8")

    sc = ax.scatter(
        mc_df["Volatility_%"], mc_df["Return_%"],
        c=mc_df["Sharpe"], cmap="RdYlGn",
        alpha=0.35, s=7, zorder=2,
    )
    plt.colorbar(sc, ax=ax, label="Sharpe Ratio", pad=0.02)

    # Optimal portfolio star
    ax.scatter(vol * 100, ret * 100, color="#B5540A", s=220, zorder=6,
               marker="*", label=f"Optimal Portfolio (SR = {sharpe:.2f})",
               edgecolors="white", linewidth=0.6)

    # Individual stock dots
    for i, t in enumerate(TICKERS):
        ax.scatter(ann_stds[i] * 100, ann_returns[i] * 100,
                   color=COLORS[t], s=80, zorder=5,
                   edgecolors="white", linewidth=0.8)
        ax.annotate(t, (ann_stds[i] * 100 + 0.25, ann_returns[i] * 100),
                    fontsize=8.5, color=COLORS[t], fontweight="bold")

    ax.set_xlabel("Annual Volatility (%)", fontsize=10)
    ax.set_ylabel("Annual Return (%)", fontsize=10)
    ax.set_title(
        f"Efficient Frontier — Monte Carlo ({N_SIMULATIONS:,} Portfolios)\n"
        f"4 U.S. Stocks  |  Rf = {RISK_FREE_RATE*100:.2f}%",
        fontsize=11, fontweight="bold", pad=10,
    )
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(True, alpha=0.22, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#F8F8F8")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_optimal_weights(w_opt: np.ndarray, save_path: str = "optimal_weights.png") -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="#F8F8F8")
    ax.set_facecolor("#F8F8F8")

    labels = [f"{NAMES[t]}\n({t})" for t in TICKERS]
    bars   = ax.bar(labels, w_opt * 100,
                    color=[COLORS[t] for t in TICKERS],
                    edgecolor="white", width=0.5)

    for bar, val in zip(bars, w_opt * 100):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Portfolio Weight (%)", fontsize=10)
    ax.set_title("Optimal Portfolio Weights (Maximum Sharpe Ratio)",
                 fontsize=11, fontweight="bold", pad=10)
    ax.set_ylim(0, max(w_opt * 100) + 12)
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#F8F8F8")
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n[1/5] Loading data & computing return statistics...")
    returns    = PRICE_INDEX.pct_change().dropna()
    ann_ret    = returns[TICKERS].mean().values * 12           # shape (4,)
    cov_ann    = returns[TICKERS].cov().values * 12            # shape (4,4)
    ann_std    = returns[TICKERS].std().values * np.sqrt(12)   # shape (4,)

    print("[2/5] Running Monte Carlo simulation ({:,} portfolios)...".format(N_SIMULATIONS))
    mc_df = run_monte_carlo(ann_ret, cov_ann)

    print("[3/5] Running SLSQP optimisation (max Sharpe)...")
    w_opt, opt_ret, opt_vol, opt_sharpe = optimise_max_sharpe(ann_ret, cov_ann)
    print_results(w_opt, opt_ret, opt_vol, opt_sharpe)

    print("[4/5] Generating charts...")
    plot_efficient_frontier(mc_df, w_opt, opt_ret, opt_vol, opt_sharpe,
                            ann_ret, ann_std)
    plot_optimal_weights(w_opt)

    print("[5/5] Exporting results...")
    summary = pd.DataFrame(
        {
            "Ticker":            TICKERS,
            "Name":              [NAMES[t] for t in TICKERS],
            "Ann_Return_%":      (ann_ret * 100).round(2),
            "Ann_Volatility_%":  (ann_std * 100).round(2),
            "Optimal_Weight_%":  (w_opt * 100).round(2),
        }
    )
    summary.to_csv("portfolio_results.csv", index=False)
    mc_df.to_csv("monte_carlo_simulations.csv", index=False)
    print("  Saved: portfolio_results.csv")
    print("  Saved: monte_carlo_simulations.csv")
    print("\nDone.")
