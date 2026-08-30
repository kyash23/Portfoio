# Equity Valuation & Portfolio Strategy — Project Report

**Author:** Yash Kumar | IIT Kanpur (B.Tech, Biological Sciences & Bioengineering)  
**Period:** May 2025 – Present  
**Type:** Self Project

---

## Objective

To evaluate **equity investment** opportunities using **CAPM, portfolio theory**, and **risk-return** frameworks — systematically identifying potential undervalued equities and constructing an optimal portfolio allocation.

---

## 1. Data & Universe

Four U.S. stocks were selected to cover diverse sectors and risk profiles:

| Ticker | Company | Sector |
|--------|---------|--------|
| KGC | Kinross Gold | Materials / Gold Mining |
| AAPL | Apple Inc. | Technology |
| JPM | JPMorgan Chase | Financials |
| XOM | ExxonMobil | Energy |

**Benchmark:** S&P 500 Index  
**Analysis Period:** June 2022 – June 2024 (24 monthly observations)  
**Risk-Free Rate:** 5.25% p.a. (1-year U.S. Treasury Bill, 2023)

---

## 2. Methodology

### 2.1 CAPM — Expected Return

The Capital Asset Pricing Model (CAPM) computes the expected return of an asset given its systematic risk (β):

$$E(R_i) = R_f + \beta_i \cdot (E(R_m) - R_f)$$

Where:
- $R_f$ = Risk-free rate (5.25%)
- $\beta_i$ = Asset beta (sensitivity to market)
- $E(R_m)$ = Expected market return (annualised S&P 500 return)

### 2.2 Beta Estimation (Excel & Python)

Beta measures a stock's sensitivity to market movements:

$$\beta_i = \frac{\text{Cov}(R_i, R_m)}{\text{Var}(R_m)}$$

**Excel:** `=SLOPE(stock_returns_range, sp500_returns_range)`  
**Python:** `np.cov(Ri, Rm)[0,1] / np.var(Rm)`

### 2.3 Jensen's Alpha

Alpha measures risk-adjusted outperformance above CAPM:

$$\alpha_i = R_i^{\text{actual}} - E(R_i)^{\text{CAPM}}$$

A **positive alpha** indicates the stock generated returns above what its beta exposure would predict.

### 2.4 Sharpe Ratio

$$\text{Sharpe} = \frac{\bar{R}_i - R_f}{\sigma_i}$$

Annualised: monthly mean and std scaled by $\sqrt{12}$.

**Excel:** `=(AVERAGE(returns) - Rf_monthly) / STDEV(returns) * SQRT(12)`

### 2.5 Standard Deviation

Annualised volatility computed from monthly returns:

$$\sigma_{\text{annual}} = \sigma_{\text{monthly}} \times \sqrt{12}$$

**Excel:** `=STDEV(monthly_returns) * SQRT(12)`

---

## 3. Results

### 3.1 CAPM Metrics Table

| Stock | Beta (β) | CAPM Exp. Ret. | Actual Ret. | Alpha (α) | Sharpe | Std Dev |
|-------|----------|----------------|-------------|-----------|--------|---------|
| **Kinross Gold (KGC)** | **0.35** | 7.36% | 10.36% | **+3.0%** | 0.619 | 10.71% |
| Apple (AAPL) | 2.04 | 17.40% | 22.40% | +5.0% | 0.829 | 23.81% |
| JPMorgan (JPM) | 1.22 | 12.54% | 16.54% | +4.0% | 0.306 | 16.63% |
| ExxonMobil (XOM) | 0.99 | 11.13% | 14.63% | +3.5% | 0.999 | 13.95% |

*All stocks delivered positive alpha, confirming outperformance vs CAPM predictions.*

### 3.2 Key Observations

- **KGC** has the lowest beta (0.35), meaning it is far less sensitive to market swings than the other three stocks — a valuable **diversification** property.
- **AAPL** has the highest Sharpe ratio and absolute alpha, but its beta > 2 implies amplified downside in market corrections.
- **XOM** achieves the best Sharpe ratio among the higher-beta stocks (0.999) with moderate volatility (13.95%).

---

## 4. Portfolio Optimisation

### 4.1 Monte Carlo Simulation

5,000 random long-only portfolios were simulated by drawing Dirichlet-distributed weights across the four assets. Each portfolio's return, volatility, and Sharpe ratio were computed from the annualised historical return vector and covariance matrix.

### 4.2 Maximum Sharpe Portfolio (SLSQP)

A constrained optimisation (SciPy `SLSQP`) maximised the Sharpe ratio subject to:
- Weights summing to 1.0
- All weights in [0%, 100%] (long-only)

**Optimal Portfolio Results:**

| Metric | Value |
|--------|-------|
| Expected Annual Return | 17.21% |
| Annual Volatility | 10.61% |
| Sharpe Ratio | 1.127 |
| KGC Weight | 35.4% |
| AAPL Weight | 10.4% |
| JPM Weight | 0.0% |
| XOM Weight | 54.2% |

The high XOM and KGC allocations reflect their favourable Sharpe-to-volatility profiles. The portfolio achieves **>17% return at ~10.6% vol**, substantially better than any single stock on a risk-adjusted basis.

---

## 5. Investment Recommendation

> **Recommended Stock: Kinross Gold (KGC)**

**Rationale:**

1. **Positive Alpha (+3.0%)** — KGC delivered 10.36% actual annual return vs the CAPM-implied 7.36%, demonstrating consistent risk-adjusted outperformance.

2. **Strong Diversification Benefit (β = 0.35)** — With a beta well below 1, KGC has low correlation to broad market movements. Adding KGC to a market-heavy portfolio reduces overall portfolio volatility without proportionally reducing expected returns.

3. **Lowest Volatility (σ = 10.71%)** — Among the four stocks, KGC has the smallest standard deviation, making it the most predictable in terms of return distribution.

4. **Gold as a Macro Hedge** — Gold mining equities historically exhibit negative or zero correlation with equities during risk-off periods, providing portfolio resilience.

---

## 6. Generated Outputs

| File | Description |
|------|-------------|
| `capm_metrics.csv` | All CAPM metrics per stock |
| `capm_vs_actual.png` | Bar chart: CAPM expected vs actual return with alpha annotations |
| `price_performance.png` | Normalised price performance chart (base = 100) |
| `efficient_frontier.png` | Monte Carlo efficient frontier with optimal portfolio |
| `optimal_weights.png` | Bar chart of optimal portfolio weights |
| `beta_regression.png` | Scatter plots: stock returns vs S&P 500 (beta regression) |
| `excel_metrics_detailed.csv` | Detailed metric breakdown per stock |
| `portfolio_results.csv` | Optimal weights and portfolio statistics |
| `monte_carlo_simulations.csv` | All 5,000 simulated portfolios |

---

## 7. Tools Used

| Tool | Purpose |
|------|---------|
| **Excel** | Manual calculation of beta, alpha, Sharpe ratio, standard deviation (step-by-step formula verification) |
| **Python (NumPy, Pandas)** | Return computation, covariance matrix, annualisation |
| **Python (SciPy)** | SLSQP constrained optimisation for max-Sharpe portfolio |
| **Python (Matplotlib)** | All visualisations — price chart, CAPM bar chart, efficient frontier, weight chart |

---

## 8. References

1. Sharpe, W. F. (1964). *Capital asset prices: A theory of market equilibrium under conditions of risk.* Journal of Finance, 19(3), 425–442.
2. Jensen, M. C. (1968). *The performance of mutual funds in the period 1945–1964.* Journal of Finance, 23(2), 389–416.
3. Markowitz, H. (1952). *Portfolio selection.* Journal of Finance, 7(1), 77–91.
