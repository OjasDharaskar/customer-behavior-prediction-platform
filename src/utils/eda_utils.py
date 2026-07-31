import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def calculate_univariate_stats(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Computes univariate statistics for the given columns in the dataframe:
    - Mean
    - Median
    - Standard Deviation
    - IQR (Interquartile Range)
    - Skewness
    """
    stats = []
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        stats.append({
            "feature": col,
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
            "iqr": float(iqr),
            "skewness": float(series.skew())
        })
    return pd.DataFrame(stats)

def calculate_vif(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Calculates the Variance Inflation Factor (VIF) for continuous predictors
    using OLS Regression.
    """
    # Filter data to keep only columns of interest and drop rows with NaNs
    df_clean = df[columns].dropna()
    
    vif_data = []
    for col in columns:
        X_other = df_clean[[c for c in columns if c != col]]
        y_col = df_clean[col]
        
        if X_other.shape[1] == 0:
            # Only one column, VIF is 1.0
            vif = 1.0
        else:
            lr = LinearRegression()
            lr.fit(X_other, y_col)
            r2 = lr.score(X_other, y_col)
            
            # Avoid division by zero if R^2 is exactly 1.0
            vif = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
            
        vif_data.append({"feature": col, "VIF": vif})
        
    return pd.DataFrame(vif_data)

def calculate_correlations(df: pd.DataFrame, feature_cols: list, target_cols: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculates linear (Pearson) and non-linear (Spearman) correlations
    between feature columns and target columns.
    """
    # Pearson correlation
    pearson_corr = df[feature_cols + target_cols].corr(method="pearson").loc[feature_cols, target_cols]
    
    # Spearman correlation
    spearman_corr = df[feature_cols + target_cols].corr(method="spearman").loc[feature_cols, target_cols]
    
    return pearson_corr, spearman_corr
