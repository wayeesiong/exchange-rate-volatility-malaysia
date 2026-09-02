# Exchange Rate Volatility and Export Growth in Malaysia

**Financial Econometrics · Python · Time-Series Analysis**

A quantitative research project examining whether MYR/USD exchange-rate volatility affects Malaysia's export growth in the short and long run. The study combines daily exchange-rate data with monthly macroeconomic indicators and applies a two-stage modelling framework: conditional volatility estimation followed by dynamic regression analysis.

> **Academic note:** This repository is based on a group assignment for FIN308 Financial Econometrics. Keep the group-project disclosure when publishing the repository, and describe individual contributions separately and accurately.

## Research objectives

- Estimate conditional MYR/USD exchange-rate volatility using an ARIMA-GARCH-family framework.
- Examine the direction of causality between exchange-rate volatility and Malaysia's export growth.
- Test the short-run relationship using an ARDL Error Correction Model (ECM).
- Test the long-run relationship using the ARDL Bounds Test approach.

## Data

The macroeconomic sample covers **January 2006 to September 2025** with **237 monthly observations**. Daily MYR/USD exchange-rate observations are used to estimate conditional volatility before aggregation to monthly frequency.

Variables used in the study include:

- MYR/USD exchange rate
- Exchange-rate volatility
- Malaysia export growth
- Real Effective Exchange Rate (REER)
- Industrial Production Index (IPI)
- Consumer Price Index (CPI)
- World Industrial Production Index (WIP)

Data sources identified in the academic report include **Bank Negara Malaysia, Department of Statistics Malaysia, International Monetary Fund, and CPB Netherlands Bureau for Economic Policy Analysis**.

## Methodology

1. **Data preparation** — prepare exchange-rate and macroeconomic time series.
2. **Volatility modelling** — estimate daily MYR/USD conditional volatility using ARIMA and GARCH-family models; the study selected **EGARCH(1,1)** as the best-fitting volatility specification.
3. **Dynamic regression** — apply the **ARDL Bounds Test** and **Error Correction Model** to distinguish long-run cointegration from short-run adjustment.
4. **Diagnostics and robustness** — apply stationarity tests, residual diagnostics, subsample stability checks, CUSUM/CUSUMSQ tests and Granger-causality analysis.

## Key findings

- The ARDL bounds-test F-statistic of **6.9242** supports a stable long-run relationship among export growth, exchange-rate volatility and the selected controls.
- Exchange-rate volatility has a **negative long-run coefficient** and is statistically significant at the 10% level in the reported model.
- The short-run volatility coefficient is negative but statistically insignificant.
- The error-correction coefficient indicates that approximately **29.2%** of a short-run deviation from long-run equilibrium is corrected within one month.
- Subsample analysis suggests the volatility-export relationship is **regime-dependent**, with a more negative relationship in 2016–2025 than in 2006–2015.
- Granger-causality testing suggests exchange-rate volatility is a secondary predictor of export growth, while world industrial production and CPI carry stronger predictive signals in the reported model.

## Repository structure

```text
exchange-rate-volatility-malaysia/
├── README.md
├── .gitignore
├── requirements.txt
├── code/
│   └── README.md
├── notebooks/
│   └── README.md
├── data/
│   ├── README.md
│   ├── raw/
│   └── processed/
├── outputs/
│   ├── README.md
│   ├── figures/
│   ├── tables/
│   └── model-results/
└── report/
    └── original-academic-report.pdf
```

## How to use this repository

1. Add the final Python scripts to `code/` and/or Jupyter notebooks to `notebooks/`.
2. Add publishable source data to `data/raw/` and transformed data to `data/processed/`.
3. Add exported charts, tables and model summaries to the relevant `outputs/` folders.
4. Review `requirements.txt` against the actual imports in the final code before publishing.
5. Remove credentials, personal file paths, student identifiers and any data that cannot legally be redistributed.

## Portfolio

This repository is intended to serve as the technical evidence layer behind the corresponding case-study page in the **Finance & Quantitative Portfolio**. Once the repository is published, link it from the portfolio page using a button such as **View Code & Data on GitHub**.
