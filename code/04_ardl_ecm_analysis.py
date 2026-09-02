from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
import seaborn as sns

from statsmodels.tsa.ardl import ARDL
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan
from statsmodels.stats.stattools import jarque_bera, durbin_watson
from statsmodels.stats.outliers_influence import reset_ramsey, variance_inflation_factor
from statsmodels.stats.diagnostic import recursive_olsresiduals
from statsmodels.tsa.stattools import grangercausalitytests

# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "analysis_dataset.csv"
)

FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "model-results"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("Reading dataset from:")
print(DATA_FILE)

df = pd.read_csv(DATA_FILE)

# ==========================================
# 1. SETUP & DATA PREPARATION
# ==========================================

target_name = 'EG'
exog_names = ['VOL', 'L_REER', 'L_WPI', 'L_CPI', 'L_IPI']
data_clean = df[[target_name] + exog_names].dropna()
y = data_clean[target_name]
X = data_clean[exog_names]

# ==========================================
# 2. OPTIMAL LAG SELECTION
# ==========================================
print("-------------------------------------------------------")
print("STEP 1: LAG SELECTION (Optimizing AIC, Max 6)")
print("-------------------------------------------------------")

max_lag = 6
current_lags = {name: 1 for name in [target_name] + exog_names}

def get_aic(lags_dict):
    try:
        exog_order = {k: v for k, v in lags_dict.items() if k != target_name}
        model = ARDL(endog=y, lags=lags_dict[target_name], exog=X, order=exog_order, trend='c')
        res = model.fit()
        return res.aic
    except:
        return float('inf')

improved = True
iteration = 0
current_aic = get_aic(current_lags)

while improved and iteration < 10:
    improved = False
    iteration += 1
    for var in [target_name] + exog_names:
        best_lag = current_lags[var]
        start_lag = 1 if var == target_name else 0
        for lag in range(start_lag, max_lag + 1):
            test_lags = current_lags.copy()
            test_lags[var] = lag
            aic = get_aic(test_lags)
            if aic < current_aic:
                current_aic = aic
                current_lags[var] = lag
                improved = True

print("Selected Lag Structure:")
for k, v in current_lags.items():
    print(f"  {k}: {v}")

# ==========================================
# 3. ESTIMATE FINAL MODEL
# ==========================================
final_exog_order = {k: v for k, v in current_lags.items() if k != target_name}
ardl_model = ARDL(endog=y, lags=current_lags[target_name], exog=X, order=final_exog_order, trend='c')
ardl_results = ardl_model.fit()
with open(
    RESULTS_DIR / "ardl_model_summary.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(ardl_results.summary().as_text())
ols_wrapper = sm.OLS(ardl_results.model._y, ardl_results.model._x).fit()

print("\n-------------------------------------------------------")
print("STEP 2: MODEL FIT STATISTICS")
print("-------------------------------------------------------")
print(f"Adjusted R-squared: {ols_wrapper.rsquared_adj:.4f}")
print(f"F-statistic:        {ols_wrapper.fvalue:.4f} (Prob: {ols_wrapper.f_pvalue:.4f})")
print(f"AIC:                {ardl_results.aic:.4f}")

model_stats = pd.DataFrame({
    "Metric": [
        "Adjusted R-squared",
        "F-statistic",
        "F-statistic p-value",
        "AIC"
    ],
    "Value": [
        ols_wrapper.rsquared_adj,
        ols_wrapper.fvalue,
        ols_wrapper.f_pvalue,
        ardl_results.aic
    ]
})

model_stats.to_csv(
    TABLE_DIR / "model_fit_statistics.csv",
    index=False
)
# ==========================================
# 4. DIAGNOSTIC TESTS
# ==========================================
print("\n-------------------------------------------------------")
print("STEP 3: DIAGNOSTIC TESTS")
print("-------------------------------------------------------")

# A. Multicollinearity
print("\n[A] Multicollinearity Tests")
print("1. Correlation Matrix:")
print(X.corr().round(4))
print("\n2. VIF:")
X_vif = sm.add_constant(X)
vif_data = pd.DataFrame()
vif_data["Variable"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
print(vif_data)

vif_data.to_csv(
    TABLE_DIR / "vif_results.csv",
    index=False
)

# B. Autocorrelation
print("\n[B] Autocorrelation Tests")
dw_stat = durbin_watson(ols_wrapper.resid)
print(f"1. Durbin-Watson: {dw_stat:.4f}")
bg_test = acorr_breusch_godfrey(ols_wrapper, nlags=2)
print(f"2. Breusch-Godfrey (Lag 2): p={bg_test[1]:.4f} -> {'PASS' if bg_test[1]>0.05 else 'FAIL'}")

# C. Heteroskedasticity
print("\n[C] Heteroskedasticity")
bp_test = het_breuschpagan(ols_wrapper.resid, ols_wrapper.model.exog)
print(f"1. Breusch-Pagan: p={bp_test[1]:.4f} -> {'PASS' if bp_test[1]>0.05 else 'FAIL'}")

plt.figure(figsize=(10, 6))
sns.scatterplot(x=ols_wrapper.fittedvalues, y=ols_wrapper.resid)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted: Check for Fan Shape (Heteroskedasticity)')
plt.savefig(
    FIGURE_DIR / "heteroskedasticity.png",
    bbox_inches="tight"
)
print("-> Heteroskedasticity Plot saved.")

# D. Normality
print("\n[D] Normality")
jb_stat, jb_pval, skew, kurt = jarque_bera(ols_wrapper.resid)
print(f"1. Jarque-Bera: p={jb_pval:.4f} -> {'PASS' if jb_pval>0.05 else 'FAIL'}")

plt.figure(figsize=(10, 5))
sns.histplot(ols_wrapper.resid, kde=True, color='blue', bins=30)
plt.title('Histogram of Residuals')
plt.savefig(
    FIGURE_DIR / "residual_histogram.png",
    bbox_inches="tight"
)
print("-> Histogram saved.")

plt.figure(figsize=(10, 5))
stats.probplot(ols_wrapper.resid, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')
plt.savefig(
    FIGURE_DIR / "residual_qq_plot.png",
    bbox_inches="tight"
)
print("-> Q-Q Plot saved.")

# E. Functional Form
print("\n[E] Functional Form")
try:
    reset_test = reset_ramsey(ols_wrapper, degree=2)
    print(f"1. Ramsey RESET: p={reset_test.pvalue:.4f} -> {'PASS' if reset_test.pvalue>0.05 else 'FAIL'}")
except: print("Error calculating RESET.")

# ==========================================
# 5. BOUNDS TEST & LONG RUN
# ==========================================
print("\n-------------------------------------------------------")
print("STEP 4: ARDL BOUNDS TEST & LONG RUN (Robust F-Stat)")
print("-------------------------------------------------------")
long_run_results = []
uecm_df = pd.DataFrame()
uecm_df['D_EG'] = y.diff()
uecm_df['EG_Lag1'] = y.shift(1)
lag_cols, diff_cols = [], []
for col in exog_names:
    uecm_df[f'{col}_Lag1'] = X[col].shift(1)
    uecm_df[f'D_{col}'] = X[col].diff()
    lag_cols.append(f'{col}_Lag1')
    diff_cols.append(f'D_{col}')
uecm_data = uecm_df.dropna()
X_uecm = sm.add_constant(uecm_data[['EG_Lag1'] + lag_cols + diff_cols])
uecm_model = sm.OLS(uecm_data['D_EG'], X_uecm).fit(cov_type='HC3')
ecm_results = pd.DataFrame({
    "Variable": uecm_model.params.index,
    "Coefficient": uecm_model.params.values,
    "Std_Error": uecm_model.bse.values,
    "t_Statistic": uecm_model.tvalues.values,
    "p_Value": uecm_model.pvalues.values
})

ecm_results.to_csv(
    TABLE_DIR / "ecm_short_run_results.csv",
    index=False
)
hypotheses = 'EG_Lag1 = 0, ' + ', '.join([f'{col} = 0' for col in lag_cols])
f_stat_bounds = uecm_model.f_test(hypotheses).fvalue
if np.ndim(f_stat_bounds) > 0: f_stat_bounds = f_stat_bounds.item()

print(f"F-Statistic: {f_stat_bounds:.4f}")
print("Bound (1%): ~4.68 | Bound (5%): ~3.79")

print("\nShort Run (ECM) Coefficients:")
print(uecm_model.summary().tables[1])
with open(
    RESULTS_DIR / "ecm_model_summary.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(uecm_model.summary().as_text())

# ==========================================
# 3. CALCULATE LONG RUN SIGNIFICANCE (The "Missing Test")
# ==========================================
print("\n-------------------------------------------------------")
print("LONG RUN COEFFICIENTS & T-STATISTICS (Robust Standard Errors)")
print("-------------------------------------------------------")

params = ardl_results.params
cov_params = ols_wrapper.cov_params()
param_names = ardl_results.model.exog_names

# 1. Identify Lag Terms for Y (Dependent) and X (Independent)
# Denominator = 1 - Sum(Beta_Y_Lags)
y_lag_indices = [i for i, name in enumerate(param_names) if target_name in name and '.L' in name]
denom = 1 - params.iloc[y_lag_indices].sum()

print(f"{'Variable':<10} | {'Coeff':<10} | {'Std.Err':<10} | {'t-Stat':<10} | {'Prob':<10}")
print("-" * 65)

for name in exog_names:
    # Numerator = Sum(Beta_X_Lags) for this variable
    x_lag_indices = [i for i, col in enumerate(param_names) if name in col]
    numerator = params.iloc[x_lag_indices].sum()
    
    # Long Run Coefficient
    lr_coeff = numerator / denom
    
    # --- DELTA METHOD FOR STANDARD ERROR ---
    # We need the derivative of (Num / Denom) with respect to EACH parameter in the model
    # d(LR)/d(Theta) = [ (1/Denom) * d(Num)/d(Theta) ] + [ (Num / Denom^2) * d(Denom)/d(Theta) ]
    
    gradient = np.zeros(len(params))
    
    # Fill gradient for X terms (Numerator part)
    # d(Num)/d(Beta_X) = 1 for lags of THIS variable, 0 otherwise
    gradient[x_lag_indices] = 1 / denom
    
    # Fill gradient for Y terms (Denominator part)
    # d(Denom)/d(Beta_Y) = -1 (because Denom = 1 - sum(Beta_Y))
    # Derivative of (Num / Denom) w.r.t Beta_Y is: Num * (-1 / Denom^2) * (-1) = Num / Denom^2
    gradient[y_lag_indices] = numerator / (denom**2)
    
    # Calculate Variance: G' * Cov * G
    lr_variance = gradient.T @ cov_params @ gradient
    lr_se = np.sqrt(lr_variance)
    
    # Calculate t-stat and p-value
    t_stat = lr_coeff / lr_se
    p_val = (1 - stats.t.cdf(abs(t_stat), df=ardl_results.nobs - len(params))) * 2
    
    # Print Row
    star = "**" if p_val < 0.05 else ""
    print(f"{name:<10} | {lr_coeff:<10.4f} | {lr_se:<10.4f} | {t_stat:<10.4f} | {p_val:<10.4f} {star}")
    
    long_run_results.append({
    "Variable": name,
    "Coefficient": lr_coeff,
    "Std_Error": lr_se,
    "t_Statistic": t_stat,
    "p_Value": p_val
    })
    
print("-" * 65)
print("(** indicates significance at 5%)")
long_run_df = pd.DataFrame(long_run_results)

long_run_df.to_csv(
    TABLE_DIR / "long_run_coefficients.csv",
    index=False
)

# ==========================================
# 6. RESIDUAL-BASED STABILITY (CUSUM & CUSUMSQ)
# ==========================================
print("\n-------------------------------------------------------")
print("STEP 5: STABILITY TESTS (CUSUM & CUSUMSQ)")
print("-------------------------------------------------------")
try:
    res_recur = recursive_olsresiduals(ols_wrapper)
    r_resid = res_recur[0] # Raw recursive residuals
    cusum = res_recur[5]   # CUSUM statistic
    
    # --- 1. CUSUM PLOT ---
    cusum_bounds = res_recur[6] 
    fig, ax = plt.subplots(figsize=(10, 6))
    min_len = min(len(cusum), len(cusum_bounds[0]))
    ax.plot(cusum[-min_len:], label='CUSUM', color='blue')
    ax.plot(cusum_bounds[0][-min_len:], 'r--', label='5% Bound')
    ax.plot(cusum_bounds[1][-min_len:], 'r--')
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_title('CUSUM Test')
    ax.legend()
    plt.savefig(
    FIGURE_DIR / "cusum.png",
    bbox_inches="tight"
    )
    print("-> CUSUM Plot saved as 'CUSUM_Plot.png'")

    # --- 2. CUSUMSQ PLOT (FIXED) ---
    # FIX: Drop NaNs (initialization period) from r_resid
    r_resid = r_resid[~np.isnan(r_resid)]
    
    # Calculate Squared CUSUM
    sigma2_t = np.cumsum(r_resid**2)
    cusumsq = sigma2_t / sigma2_t[-1]
    
    # Bounds calculation (Approx 5%: +/- 1.358/sqrt(T))
    T = len(cusumsq)
    t = np.linspace(0, 1, T)
    c0 = 1.358 / np.sqrt(T)
    upper_bound = t + c0
    lower_bound = t - c0
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(t, cusumsq, label='CUSUMSQ', color='green')
    ax2.plot(t, upper_bound, 'r--', label='5% Bound')
    ax2.plot(t, lower_bound, 'r--')
    ax2.set_ylim(-0.1, 1.1) 
    ax2.plot(t, t, 'k--', alpha=0.3)
    ax2.set_title('CUSUMSQ Test')
    ax2.legend()
    plt.savefig(
    FIGURE_DIR / "cusumsq.png",
    bbox_inches="tight"
    )
    print("-> CUSUMSQ Plot saved as 'CUSUMSQ_Plot.png'")

except Exception as e:
    print(f"Stability Error: {e}")


# ==========================================
# 7. SUBSAMPLE STABILITY TEST ("Safe Mode")
# ==========================================
print("\n-------------------------------------------------------")
print("STEP 6: SUBSAMPLE STABILITY TEST (Robustness Check)")
print("-------------------------------------------------------")
split_idx = len(data_clean) // 2
sub1 = data_clean.iloc[:split_idx]
sub2 = data_clean.iloc[split_idx:]

def get_safe_lr(data_sub):
    try:
        y_s, X_s = data_sub[target_name], data_sub[exog_names]
        simple_order = {n: 2 for n in exog_names}
        model = ARDL(endog=y_s, lags=2, exog=X_s, order=simple_order, trend='c')
        res = model.fit()
        p = res.params
        den = 1 - sum([v for k,v in p.items() if target_name in k and 'L' in k])
        return {n: sum([v for k,v in p.items() if n in k])/den for n in exog_names}
    except: return {}

lr1 = get_safe_lr(sub1)
lr2 = get_safe_lr(sub2)

subsample_results = pd.DataFrame({
    "Variable": exog_names,
    "Subsample_1": [
        lr1.get(n, np.nan) for n in exog_names
    ],
    "Subsample_2": [
        lr2.get(n, np.nan) for n in exog_names
    ]
})

subsample_results.to_csv(
    TABLE_DIR / "subsample_stability.csv",
    index=False
)

print(f"{'Variable':<10} | {'Subsample 1':<12} | {'Subsample 2':<12}")
print("-" * 40)
for n in exog_names:
    val1 = lr1.get(n, np.nan)
    val2 = lr2.get(n, np.nan)
    print(f"{n:<10} | {val1:<12.4f} | {val2:<12.4f}")


# ==========================================
# 8. GRANGER CAUSALITY TEST
# ==========================================
print("\n-------------------------------------------------------")
print("STEP 7: GRANGER CAUSALITY TEST")
print("-------------------------------------------------------")
gc_df = pd.DataFrame()
gc_df['EG'] = y
gc_df['VOL'] = X['VOL']
gc_df['D_L_REER'] = X['L_REER'].diff()
gc_df['D_L_WPI'] = X['L_WPI'].diff()
gc_df['D_L_CPI'] = X['L_CPI'].diff()
gc_df['D_L_IPI'] = X['L_IPI'].diff()
gc_clean = gc_df.dropna()

preds = ['VOL', 'D_L_REER', 'D_L_WPI', 'D_L_CPI', 'D_L_IPI']
print(f"{'Direction':<20} | {'p-value':<10} | {'Result'}")
granger_results = []
for p in preds:
    res = grangercausalitytests(gc_clean[['EG', p]], maxlag=[2], verbose=False)
    pval = res[2][0]['ssr_ftest'][1]
    print(f"{p:<12} -> EG  | {pval:.4f}     | {'Causal' if pval<0.05 else 'No Cause'}")
    granger_results.append({
    "Predictor": p,
    "Direction": f"{p} -> EG",
    "Lag": 2,
    "p_Value": pval,
    "Result": "Causal" if pval < 0.05 else "No Cause"
    })
    
granger_df = pd.DataFrame(granger_results)

granger_df.to_csv(
    TABLE_DIR / "granger_causality.csv",
    index=False
)