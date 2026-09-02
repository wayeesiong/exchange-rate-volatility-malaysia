from pathlib import Path
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from arch.unitroot import PhillipsPerron

PROJECT_ROOT = Path(__file__).resolve().parents[1]

input_file = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "analysis_dataset.csv"
)

print("Reading:", input_file)

df = pd.read_csv(input_file)


# Define the list of variables to test
variables_to_test = ['EG', 'VOL', 'L_REER', 'L_WPI', 'L_CPI', 'L_IPI']

# ==========================================
# 2. DEFINE THE TEST FUNCTIONS
# ==========================================

def run_adf(series, name):
    """
    Runs ADF test at Level and First Difference.
    """
    series = series.dropna()
    
    # --- Test at Level ---
    adf_level = adfuller(series, autolag='AIC', regression='c')
    
    # --- Test at First Difference ---
    diff_series = series.diff().dropna()
    adf_diff = adfuller(diff_series, autolag='AIC', regression='c')
    
    # Determine Integration Order
    if adf_level[1] < 0.05:
        order = "I(0)"
    elif adf_diff[1] < 0.05:
        order = "I(1)"
    else:
        order = "I(2) or higher (FAIL)"

    return {
        'Variable': name,
        'ADF Level (t-stat)': round(adf_level[0], 3),
        'ADF Level (p-val)': round(adf_level[1], 3),  # Rounded to 3 decimals
        'ADF Diff (t-stat)': round(adf_diff[0], 3),
        'ADF Diff (p-val)': round(adf_diff[1], 3),    # Rounded to 3 decimals
        'Result': order
    }

def run_pp(series, name):
    """
    Runs Phillips-Perron test using the 'arch' library.
    """
    series = series.dropna()
    
    # --- Test at Level ---
    pp_level = PhillipsPerron(series, trend='c', test_type='tau')
    
    # --- Test at First Difference ---
    diff_series = series.diff().dropna()
    pp_diff = PhillipsPerron(diff_series, trend='c', test_type='tau')
    
    # Determine Integration Order
    if pp_level.pvalue < 0.05:
        order = "I(0)"
    elif pp_diff.pvalue < 0.05:
        order = "I(1)"
    else:
        order = "I(2) or higher"

    return {
        'Variable': name,
        'PP Level (t-stat)': round(pp_level.stat, 3),
        'PP Level (p-val)': round(pp_level.pvalue, 3), # Rounded to 3 decimals
        'PP Diff (t-stat)': round(pp_diff.stat, 3),
        'PP Diff (p-val)': round(pp_diff.pvalue, 3),   # Rounded to 3 decimals
        'Result': order
    }

# ==========================================
# 3. RUN TESTS AND PRINT RESULTS
# ==========================================

adf_results = []
pp_results = []

print("Running Stationarity Tests...\n")

for var in variables_to_test:
    if var in df.columns:
        adf_results.append(run_adf(df[var], var))
        pp_results.append(run_pp(df[var], var))
    else:
        print(f"Warning: Variable '{var}' not found. Did you forget to log it?")

# Convert to Pandas DataFrames
adf_table = pd.DataFrame(adf_results)
pp_table = pd.DataFrame(pp_results)

print("--- AUGMENTED DICKEY-FULLER (ADF) TEST RESULTS ---")
print(adf_table.to_string(index=False)) # to_string makes it print nicely
print("\n")

print("--- PHILLIPS-PERRON (PP) TEST RESULTS ---")
print(pp_table.to_string(index=False))
print("\n")

# ==========================================
# 4. EXPORT TO CSV
# ==========================================
# Using a generic name, or you can add your path here too
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

adf_table.to_csv(OUTPUT_DIR / "adf_results.csv", index=False)
pp_table.to_csv(OUTPUT_DIR / "pp_results.csv", index=False)

print("Results saved to:", OUTPUT_DIR)