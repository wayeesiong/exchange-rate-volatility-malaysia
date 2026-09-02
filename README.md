# Exchange Rate Volatility and Export Growth in Malaysia

**Python | Financial Econometrics | ARIMA | EGARCH | ARDL | ECM**

## Overview

This project investigates the impact of MYR/USD exchange-rate volatility on Malaysia's export growth using Python-based time-series econometric modelling.

The analysis combines volatility modelling with macroeconomic time-series analysis to examine whether exchange-rate volatility affects export performance differently in the short run and long run.

## Research Objectives

- Estimate MYR/USD exchange-rate volatility using ARIMA and EGARCH models.
- Examine the long-run relationship between exchange-rate volatility and Malaysia's export growth.
- Evaluate short-run adjustment dynamics using an Error Correction Model.
- Conduct diagnostic, stability, and causality tests to assess model robustness.

## Methodology

### 1. Exchange Rate Volatility Estimation
Daily MYR/USD log returns are analysed using:

- ARIMA model selection
- GARCH(1,1)
- EGARCH(1,1)
- AIC and BIC model comparison

The estimated conditional volatility is subsequently converted to monthly frequency for macroeconomic analysis.

### 2. Stationarity Testing

The following tests are applied:

- Augmented Dickey-Fuller (ADF)
- Phillips-Perron (PP)

### 3. ARDL and ECM Analysis

An Autoregressive Distributed Lag model is used to evaluate the relationship between export growth and:

- Exchange-rate volatility
- Real Effective Exchange Rate
- World Industrial Production Index
- Consumer Price Index
- Industrial Production Index

The analysis includes:

- Optimal lag selection
- ARDL Bounds Test
- Long-run coefficient estimation
- Error Correction Model
- Diagnostic tests
- CUSUM and CUSUMSQ stability tests
- Subsample stability testing
- Granger causality testing

## Key Findings
1. The selected volatility specification captures substantial persistence in MYR/USD exchange-rate volatility.
2. Evidence indicates a long-run relationship between exchange-rate volatility and Malaysia's export growth.
3. Exchange-rate volatility exhibits a negative long-run association with export performance.
4. The short-run effect of exchange-rate volatility is weaker than the long-run effect.
5. Stability tests suggest that the relationship varies across major macroeconomic regimes and periods of economic stress.

## Tools & Libraries

- Python
- pandas
- NumPy
- statsmodels
- arch
- pmdarima
- SciPy
- Matplotlib
- Seaborn

## Repository Structure

```text
code/
├── 01_arima_egarch.py
├── 02_monthly_volatility.py
├── 03_stationarity_tests.py
└── 04_ardl_ecm_analysis.py

data/
├── raw/
└── processed/

outputs/
├── figures/
├── tables/
└── model-results/

report/
└── exchange-rate-volatility-malaysia.pdf
