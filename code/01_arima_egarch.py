# -*- coding: utf-8 -*-
"""
MYR/USD Volatility Analysis (ARIMA-GARCH/EGARCH)
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA = PROJECT_ROOT / "data" / "raw" / "myr_usd_log_returns.csv"
DAILY_OUTPUT = PROJECT_ROOT / "data" / "processed" / "daily_volatility.csv"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pmdarima as pm
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Suppress warnings for cleaner output in Spyder console
warnings.filterwarnings("ignore")

# =============================================================================
# 1. DATA LOADING & PREPARATION
# =============================================================================
# TODO: Update 'your_dataset.csv' with your actual filename
file_path = RAW_DATA

# Read CSV. Assuming columns are something like 'Date' and 'LogReturn'
# Update 'Date' and 'LogReturn' below to match your exact column headers
try:
    df = pd.read_csv(file_path)
    
    # Convert Date column to datetime objects
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Set Date as index
    df.set_index('Date', inplace=True)
    
    # Sort by date to ensure chronological order
    df.sort_index(inplace=True)
    
    # Filter for your specific period if needed (Jan 2006 - Sep 2025)
    #df = df['2006-01-01':'2025-09-30']
    
    # Identify the column with log returns
    # CHANGE THIS if your column name is different (e.g., 'Return', 'Close', etc.)
    return_col = 'LogReturn' 
    
    # Remove any NaN or Infinite values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    print(f"✅ Data Loaded successfully. Observations: {len(df)}")
    print(f"   Date Range: {df.index.min().date()} to {df.index.max().date()}")

except Exception as e:
    print(f"❌ Error Loading Data: {e}")
    print("   Please check your CSV filename and column headers.")
    raise SystemExit

# SCALE CHECK: GARCH optimizers fail if numbers are too small (e.g., 0.0001).
# We want returns in percentages (e.g., 0.5 for 0.5%).
# If your variance is very small (< 0.1), we likely need to multiply by 100.
if df[return_col].var() < 0.1:
    print("ℹ️  Scaling Log Returns by 100 for GARCH stability...")
    df['Scaled_Return'] = df[return_col] * 100
else:
    df['Scaled_Return'] = df[return_col]

# =============================================================================
# 2. FIND BEST ARIMA MODEL (The Mean Equation)
# =============================================================================
print("\n" + "="*60)
print("STEP 1: Finding Best ARIMA(p,d,q) Model")
print("="*60)

# auto_arima will try different combinations to find the lowest AIC
# stationary=True because Log Returns are already stationary
best_arima = pm.auto_arima(df['Scaled_Return'],
                           start_p=0, start_q=0,
                           max_p=5, max_q=5,
                           d=0,              # d=0 because returns are already differenced
                           seasonal=False,   # Exchange rates are usually non-seasonal
                           stepwise=True,
                           suppress_warnings=True,
                           trace=True)       # Prints progress

print(f"\n🏆 Optimal ARIMA Order found: {best_arima.order}")

# Fit the best ARIMA model manually to extract clean residuals
# We use statsmodels ARIMA here for easier integration
p, d, q = best_arima.order
arima_model = ARIMA(df['Scaled_Return'], order=(p, d, q))
arima_result = arima_model.fit()

# Save residuals for GARCH
df['Residuals'] = arima_result.resid

# =============================================================================
# 3. COMPARE GARCH VS EGARCH (The Variance Equation)
# =============================================================================
print("\n" + "="*60)
print("STEP 2: Fitting Volatility Models (GARCH vs EGARCH)")
print("="*60)

# --- Model A: Standard GARCH(1,1) ---
# This assumes symmetry (Good news = Bad news)
garch_spec = arch_model(df['Residuals'], vol='Garch', p=1, q=1, dist='Normal')
garch_res = garch_spec.fit(disp='off')

# --- Model B: EGARCH(1,1) ---
# This assumes asymmetry (Leverage Effect: Bad news > Good news)
# o=1 allows for the asymmetric term
egarch_spec = arch_model(df['Residuals'], vol='EGARCH', p=1, q=1, o=1, dist='Normal')
egarch_res = egarch_spec.fit(disp='off')

# --- Compare Results ---
print("\n--- MODEL COMPARISON ---")
print(f"{'Model':<15} | {'AIC':<12} | {'BIC':<12} | {'Log-Likelihood'}")
print("-" * 55)
print(f"{'GARCH(1,1)':<15} | {garch_res.aic:<12.4f} | {garch_res.bic:<12.4f} | {garch_res.loglikelihood:.4f}")
print(f"{'EGARCH(1,1)':<15} | {egarch_res.aic:<12.4f} | {egarch_res.bic:<12.4f} | {egarch_res.loglikelihood:.4f}")

# Determine the winner
if egarch_res.aic < garch_res.aic:
    best_model = egarch_res
    best_name = "EGARCH(1,1)"
    print("\n✅ WINNER: EGARCH(1,1)")
    print("   Reason: Lower AIC indicates better fit.")
    print("   implication: Asymmetric leverage effect is likely present.")
else:
    best_model = garch_res
    best_name = "GARCH(1,1)"
    print("\n✅ WINNER: GARCH(1,1)")
    print("   Reason: Lower/Similar AIC suggests simpler model is sufficient.")

print("\n--- Best Model Summary ---")
print(best_model.summary())

# =============================================================================
# 4. EXPORT RESULTS
# =============================================================================
# Extract Conditional Volatility (Conditional Variance ^ 0.5)
df['Conditional_Volatility'] = best_model.conditional_volatility

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Conditional_Volatility'], color='#d62728', linewidth=0.8)
plt.title(f'MYR/USD Exchange Rate Volatility ({best_name})', fontsize=14)
plt.ylabel('Volatility (%)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Save to CSV
output_filename = DAILY_OUTPUT
df[['Scaled_Return', 'Conditional_Volatility']].to_csv(output_filename)
print(f"\n💾 Results saved to: {output_filename}")
print("   Use column 'Conditional_Volatility' for your regression analysis.")

"""
MYR/USD Volatility Estimation
Model: ARIMA(0,0,0) - EGARCH(1,1)
"""


###############################################################################


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# =============================================================================
# 1. LOAD & PREPARE DATA
# =============================================================================
# TODO: Replace with your actual filename
file_path = RAW_DATA

try:
    # Load data (Assuming columns 'Date' and 'LogReturn')
    # Adjust 'LogReturn' if your column name is different
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    
    # Filter timeframe if needed
    df = df['2006-01-01':'2025-09-30']
    
    # Identify return column
    return_col = 'LogReturn' 
    
    # Clean data (remove NaNs/Infs)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    # SCALE CHECK:
    # GARCH models optimize better when returns are in percentages (e.g. 1.0 instead of 0.01)
    # If variance is tiny (< 0.1), we multiply by 100.
    if df[return_col].var() < 0.1:
        print("ℹ️  Scaling Log Returns by 100 for better optimization...")
        df['Scaled_Return'] = df[return_col] * 100
    else:
        df['Scaled_Return'] = df[return_col]

    print(f"✅ Data Loaded. {len(df)} observations.")

except Exception as e:
    print(f"❌ Error: {e}")
    raise SystemExit

# =============================================================================
# 2. MODEL ESTIMATION: ARIMA(0,0,0) - EGARCH(1,1)
# =============================================================================
print("\n" + "="*60)
print("ESTIMATING MODEL: Constant Mean - EGARCH(1,1)")
print("="*60)

# ARIMA(0,0,0) implies just a constant mean (or zero mean).
# We use mean='Constant' to allow for an intercept (mu).
# If you prefer strictly zero mean, change to mean='Zero'.
model = arch_model(df['Scaled_Return'],
                   mean='Constant',  # Corresponds to ARIMA(0,0,0) with intercept
                   vol='EGARCH',     # The chosen volatility model
                   p=1,              # Lag order for GARCH term
                   o=1,              # Lag order for Asymmetric term
                   q=1,              # Lag order for ARCH term
                   dist='t')    # Distribution of residuals

# Fit the model
# disp='off' hides the iteration logs
res = model.fit(disp='off')

# Print Summary
print(res.summary())

# =============================================================================
# 3. EXTRACT VOLATILITY & PLOT
# =============================================================================
# conditional_volatility is the standard deviation (sigma)
df['Conditional_Volatility'] = res.conditional_volatility

# 1. Plot the Volatility
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Conditional_Volatility'], color='darkred', linewidth=0.8)
plt.title('Estimated Conditional Volatility (EGARCH 1,1)', fontsize=14)
plt.ylabel('Volatility (%)', fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 2. Save Results to CSV
# This file will be the input for your ARDL regression later
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

output_filename = OUTPUT_DIR / "daily_volatility.csv"

df[['Scaled_Return', 'Conditional_Volatility']].to_csv(
    output_filename
)

print("\n" + "=" * 60)
print(f"✅ SUCCESS: Volatility data saved to '{output_filename}'")
print("=" * 60)


###############################################################################


# ... (Run this after fitting your 'res' model from the previous code) ...

from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

# 1. Get Standardized Residuals (Residuals / Volatility)
std_resid = res.resid / res.conditional_volatility

# 2. Ljung-Box Test (Checks for remaining autocorrelation)
# We check at lag 10 (standard for daily data)
lb_test = acorr_ljungbox(std_resid, lags=[10], return_df=True)
print("\n=== DIAGNOSTIC 1: Ljung-Box Test ===")
print(lb_test)

# 3. ARCH-LM Test (Checks for remaining ARCH effects)
# We check at lag 5
lm_test = het_arch(std_resid, ddof=4)
print("\n=== DIAGNOSTIC 2: ARCH-LM Test ===")
print(f"LM Statistic: {lm_test[0]:.4f}")
print(f"P-value:      {lm_test[1]:.4f}")

# 4. Descriptive Statistics for Volatility (For the next section)
print("\n=== DESCRIPTIVE STATS (Conditional Volatility) ===")
print(df['Conditional_Volatility'].describe())