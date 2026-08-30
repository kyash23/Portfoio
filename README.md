# Equity Valuation & Portfolio Strategy

> Self Project | Yash Kumar, IIT Kanpur | May 2025 – Present

Evaluated equity investment opportunities across **4 U.S. stocks** using **CAPM, portfolio theory**, and **risk-return** frameworks. Computed alpha, beta, Sharpe ratio, and standard deviation in Excel; used Python for return simulations, portfolio optimisation, and graphical diagnostics. Recommended **Kinross Gold (KGC)** based on a **positive alpha of +3.0%** and strong diversification benefits.

---

## Results at a Glance

| Stock | Beta (β) | CAPM Exp. Ret. | Actual Ret. | Alpha (α) | Sharpe | Std Dev |
|-------|:--------:|:--------------:|:-----------:|:---------:|:------:|:-------:|
| **Kinross Gold (KGC)** ⭐ | **0.35** | 7.36% | 10.36% | **+3.0%** | 0.619 | 10.71% |
| Apple (AAPL) | 2.04 | 17.40% | 22.40% | +5.0% | 0.829 | 23.81% |
| JPMorgan (JPM) | 1.22 | 12.54% | 16.54% | +4.0% | 0.306 | 16.63% |
| ExxonMobil (XOM) | 0.99 | 11.13% | 14.63% | +3.5% | 0.999 | 13.95% |

**Optimal Portfolio (Max Sharpe):** Return = 17.21% | Volatility = 10.61% | Sharpe = 1.127

---

## Repository Structure

```
equity-valuation-portfolio/
│
├── capm_analysis.py          # CAPM metrics: beta, alpha, Sharpe, std dev
├── portfolio_optimization.py # Monte Carlo + SLSQP max-Sharpe optimisation
├── excel_metrics.py          # Step-by-step Excel formula walkthrough
│
├── REPORT.md                 # Full written project report
├── requirements.txt
└── README.md
```

---

## Methodology

### 1 · CAPM — Expected Return

$$E(R_i) = R_f + \beta_i \cdot (E(R_m) - R_f)$$

- **Rf = 5.25%** (1-yr U.S. T-bill, 2023)  
- Beta estimated via OLS regression of monthly stock returns on S&P 500 returns  
- Corresponds to Excel `=SLOPE()` function

### 2 · Jensen's Alpha

$$\alpha_i = R_i^{\text{actual}} - E(R_i)^{\text{CAPM}}$$

Alpha > 0 means the stock outperformed its CAPM-implied expected return on a risk-adjusted basis.

### 3 · Sharpe Ratio & Standard Deviation

$$\text{Sharpe} = \frac{\bar{R}_i - R_f}{\sigma_i} \qquad \sigma_{\text{annual}} = \sigma_{\text{monthly}} \times \sqrt{12}$$

Computed in **Excel** using `=STDEV()` and `=AVERAGE()`, then replicated and verified in Python.

### 4 · Portfolio Optimisation

- **Monte Carlo:** 5,000 random long-only portfolios drawn from a Dirichlet distribution
- **SLSQP:** Constrained optimisation (SciPy) maximising the Sharpe ratio, with weights ∈ [0, 1] and Σwᵢ = 1

---

## Quickstart

```bash
# Clone & install
git clone https://github.com/<your-username>/equity-valuation-portfolio.git
cd equity-valuation-portfolio
pip install -r requirements.txt

# Run CAPM analysis (prints table + saves charts + CSV)
python capm_analysis.py

# Run portfolio optimisation (efficient frontier + optimal weights)
python portfolio_optimization.py

# Step-by-step Excel formula walkthrough
python excel_metrics.py
```

---

## Generated Outputs

Running all three scripts produces:

| File | Description |
|------|-------------|
| `price_performance.png` | Normalised price chart, all 4 stocks vs S&P 500 |
| `capm_vs_actual.png` | CAPM expected vs actual return, alpha annotated |
| `efficient_frontier.png` | Monte Carlo frontier + optimal portfolio star |
| `optimal_weights.png` | Optimal portfolio weight allocation |
| `beta_regression.png` | Beta OLS regression scatter (each stock vs S&P 500) |
| `capm_metrics.csv` | All CAPM metrics per stock |
| `portfolio_results.csv` | Optimal weights and portfolio-level statistics |
| `monte_carlo_simulations.csv` | All 5,000 simulated portfolios |
| `excel_metrics_detailed.csv` | Verbose metric breakdown matching Excel steps |

---

## Recommendation

**Kinross Gold (KGC)** was selected as the top pick on two grounds:

1. **Positive Alpha (+3.0%)** — Delivered 10.36% vs the CAPM-implied 7.36%, indicating consistent risk-adjusted outperformance relative to its market exposure.

2. **Diversification Benefit (β = 0.35)** — The lowest beta in the universe means KGC has low correlation with broad market moves. It pulls overall portfolio volatility down without proportionally reducing expected returns, which is why the optimal max-Sharpe portfolio allocates 35.4% to KGC.

---

## Tools & Libraries

| Tool | Use |
|------|-----|
| Excel | Manual alpha, beta, Sharpe, std dev computation (formula verification) |
| Python · NumPy & Pandas | Return computation, covariance matrix, annualisation |
| Python · SciPy | SLSQP constrained optimisation |
| Python · Matplotlib | All charts and visualisations |

---

## References

- Sharpe, W. F. (1964). Capital asset prices. *Journal of Finance*, 19(3), 425–442.  
- Jensen, M. C. (1968). Performance of mutual funds. *Journal of Finance*, 23(2), 389–416.  
- Markowitz, H. (1952). Portfolio selection. *Journal of Finance*, 7(1), 77–91.
