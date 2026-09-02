# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 16:35:23 2025

@author: wayee
"""

# -*- coding: utf-8 -*-
"""
Convert Daily GARCH Volatility to Monthly Average for ARDL
"""

from pathlib import Path
import pandas as pd

# Project root = exchange-rate-volatility-malaysia/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Input and output paths
input_file = PROJECT_ROOT / "data" / "processed" / "daily_volatility.csv"
output_file = PROJECT_ROOT / "data" / "processed" / "monthly_volatility.csv"

print("Reading:", input_file)

# Load daily volatility data
df = pd.read_csv(input_file)

df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)
df.sort_index(inplace=True)

# Convert daily conditional volatility to monthly average
monthly_df = (
    df[["Conditional_Volatility"]]
    .resample("ME")
    .mean()
)

monthly_df.rename(
    columns={"Conditional_Volatility": "Monthly_Volatility_Avg"},
    inplace=True
)

monthly_df.to_csv(output_file)

print("\nConversion complete.")
print("Saved to:", output_file)
print("Total months:", len(monthly_df))
print(monthly_df.head())

# ==========================================
# 2. RESAMPLE TO MONTHLY FREQUENCY
# ==========================================
# 'M' stands for Month End. 
# We calculate the MEAN of the daily volatilities for each month.
# This represents the "Average Daily Volatility" experienced during that month.

monthly_df = df[['Conditional_Volatility']].resample('ME').mean()

# Rename the column to indicate it's monthly
monthly_df.rename(columns={'Conditional_Volatility': 'Monthly_Volatility_Avg'}, inplace=True)

# ==========================================
# 3. SAVE TO CSV
# ==========================================
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

output_file = OUTPUT_DIR / "monthly_volatility.csv"
monthly_df.to_csv(output_file)

print("\n" + "=" * 50)
print("CONVERSION COMPLETE")
print("=" * 50)
print(f"Saved Monthly Data to: {output_file}")
print(f"Total Months: {len(monthly_df)}")
print("\nFirst 5 Months of Data:")
print(monthly_df.head())