import pandas as pd
import numpy as np

def generate_features(clean_path: str, returns_path: str, cutoff_date: str = "2011-09-10", save_path: str = None) -> pd.DataFrame:
    """
    Generates domain-specific features and targets for customer behavior prediction using point-in-time cutoff.
    
    Args:
        clean_path (str): Path to cleaned interim transactions CSV.
        returns_path (str): Path to returned transactions CSV.
        cutoff_date (str): Cutoff date for splitting history and future targets. Default is "2011-09-10".
        save_path (str): Path to save the output processed CSV. Default is None.
        
    Returns:
        pd.DataFrame: Processed customer-level dataset containing RFM, behavioral features, and targets.
    """
    # Load data
    df_clean = pd.read_csv(clean_path)
    df_returns = pd.read_csv(returns_path)
    
    # Convert dates to datetime objects
    df_clean["invoice_date"] = pd.to_datetime(df_clean["invoice_date"])
    df_returns["invoice_date"] = pd.to_datetime(df_returns["invoice_date"])
    
    cutoff_dt = pd.to_datetime(cutoff_date)
    target_dt_end = cutoff_dt + pd.Timedelta(days=90)
    
    # Filter features: historical data up to and including cutoff
    df_feat_clean = df_clean[df_clean["invoice_date"] <= cutoff_dt].copy()
    df_feat_returns = df_returns[df_returns["invoice_date"] <= cutoff_dt].copy()
    
    # Filter targets: future transactions within the 90-day target window
    df_targ_clean = df_clean[(df_clean["invoice_date"] > cutoff_dt) & (df_clean["invoice_date"] <= target_dt_end)].copy()
    
    # Identify customers active prior to cutoff
    active_customers = df_feat_clean["customer_id"].dropna().unique()
    
    # Initialize features DataFrame
    features_df = pd.DataFrame({"customer_id": active_customers})
    
    # Group historical clean transactions by customer
    cust_clean_grouped = df_feat_clean.groupby("customer_id")
    
    # --- RFM FEATURES ---
    # 1. Recency: Days since last purchase prior to cutoff
    last_purchase_dates = cust_clean_grouped["invoice_date"].max()
    features_df["recency"] = features_df["customer_id"].map(
        lambda cid: (cutoff_dt - last_purchase_dates.get(cid)).days if cid in last_purchase_dates else np.nan
    )
    
    # 2. Frequency: Number of unique invoices prior to cutoff
    invoices_per_cust = cust_clean_grouped["invoice_no"].nunique()
    features_df["frequency"] = features_df["customer_id"].map(invoices_per_cust)
    
    # 3. Monetary Value (total and avg spend)
    df_feat_clean["total_spend"] = df_feat_clean["quantity"] * df_feat_clean["unit_price"]
    spend_per_cust = cust_clean_grouped["total_spend"].sum()
    features_df["monetary_total"] = features_df["customer_id"].map(spend_per_cust)
    features_df["monetary_avg"] = features_df["monetary_total"] / features_df["frequency"]
    
    # --- BEHAVIORAL FEATURES ---
    # 4. Average Basket Size (total quantity per unique invoice)
    quantity_per_cust = cust_clean_grouped["quantity"].sum()
    features_df["avg_basket_size"] = features_df["customer_id"].map(quantity_per_cust) / features_df["frequency"]
    
    # 5. Purchase Frequency Trend (slope / difference)
    # Difference in invoice frequency between last 90 days and the 90-180 days prior
    t_90 = cutoff_dt - pd.Timedelta(days=90)
    t_180 = cutoff_dt - pd.Timedelta(days=180)
    
    invoices_last_90d = df_feat_clean[df_feat_clean["invoice_date"] > t_90].groupby("customer_id")["invoice_no"].nunique()
    invoices_prior_90d = df_feat_clean[(df_feat_clean["invoice_date"] > t_180) & (df_feat_clean["invoice_date"] <= t_90)].groupby("customer_id")["invoice_no"].nunique()
    
    features_df["frequency_last_90d"] = features_df["customer_id"].map(invoices_last_90d).fillna(0)
    features_df["frequency_prior_90d"] = features_df["customer_id"].map(invoices_prior_90d).fillna(0)
    features_df["purchase_frequency_trend"] = features_df["frequency_last_90d"] - features_df["frequency_prior_90d"]
    
    # 6. Variance of purchase intervals
    invoice_dates_df = df_feat_clean.groupby(["customer_id", "invoice_no"])["invoice_date"].min().reset_index().sort_values(["customer_id", "invoice_date"])
    invoice_dates_df["prev_date"] = invoice_dates_df.groupby("customer_id")["invoice_date"].shift(1)
    invoice_dates_df["interval"] = (invoice_dates_df["invoice_date"] - invoice_dates_df["prev_date"]).dt.days
    interval_vars = invoice_dates_df.groupby("customer_id")["interval"].var().fillna(0.0)
    features_df["variance_purchase_intervals"] = features_df["customer_id"].map(interval_vars).fillna(0.0)
    
    # 7. Seasonal concentration: proportion of purchases in Q4 (Oct, Nov, Dec)
    invoice_months = df_feat_clean.groupby(["customer_id", "invoice_no"])["invoice_date"].min().reset_index()
    invoice_months["is_q4"] = invoice_months["invoice_date"].dt.month.isin([10, 11, 12]).astype(int)
    q4_invoices_per_cust = invoice_months.groupby("customer_id")["is_q4"].sum()
    features_df["seasonal_concentration"] = features_df["customer_id"].map(q4_invoices_per_cust).fillna(0.0) / features_df["frequency"]
    
    # 8. Return rate: total returned items / (purchased items + returned items)
    returns_per_cust = df_feat_returns.groupby("customer_id")["quantity"].sum().abs()
    purchased_per_cust = df_feat_clean.groupby("customer_id")["quantity"].sum()
    
    features_df["total_purchased_qty"] = features_df["customer_id"].map(purchased_per_cust).fillna(0.0)
    features_df["total_returned_qty"] = features_df["customer_id"].map(returns_per_cust).fillna(0.0)
    features_df["return_rate"] = features_df["total_returned_qty"] / (features_df["total_purchased_qty"] + features_df["total_returned_qty"] + 1e-5)
    
    # 9. Markdown/discount sensitivity
    clean_discounts = df_feat_clean[df_feat_clean["stock_code"] == "D"].groupby("customer_id")["invoice_no"].count()
    return_discounts = df_feat_returns[df_feat_returns["stock_code"] == "D"].groupby("customer_id")["invoice_no"].count()
    features_df["discount_count"] = features_df["customer_id"].map(clean_discounts).fillna(0.0) + features_df["customer_id"].map(return_discounts).fillna(0.0)
    features_df["discount_sensitivity"] = features_df["discount_count"] / features_df["frequency"]
    
    # 10. Customer Lifetime Value: historical CLV (monetary total)
    features_df["historical_clv"] = features_df["monetary_total"]
    
    # --- OUTPUTS & REPRESENTATIONS ---
    # 11. Product category affinity vector for top 10 stock codes
    top_10_codes = ['85123A', '22423', '85099B', '20725', '21212', '84879', '21232', '47566', '22197', '22383']
    for code in top_10_codes:
        code_counts = df_feat_clean[df_feat_clean["stock_code"] == code].groupby("customer_id")["quantity"].sum()
        features_df[f"affinity_{code}"] = features_df["customer_id"].map(code_counts).fillna(0.0) / (features_df["total_purchased_qty"] + 1e-5)
        
    # 12. Composite Engagement Score (scaled combination of RFM)
    rec_min, rec_max = features_df["recency"].min(), features_df["recency"].max()
    freq_min, freq_max = features_df["frequency"].min(), features_df["frequency"].max()
    mon_min, mon_max = features_df["monetary_total"].min(), features_df["monetary_total"].max()
    
    rec_norm = 1.0 - (features_df["recency"] - rec_min) / (rec_max - rec_min + 1e-5)
    freq_norm = (features_df["frequency"] - freq_min) / (freq_max - freq_min + 1e-5)
    mon_norm = (features_df["monetary_total"] - mon_min) / (mon_max - mon_min + 1e-5)
    
    features_df["engagement_score"] = 0.4 * rec_norm + 0.3 * freq_norm + 0.3 * mon_norm
    
    # --- TARGETS ---
    # Target 1: Churn Label (1 if NO purchase in the next 90 days, else 0)
    targ_customers = df_targ_clean["customer_id"].dropna().unique()
    features_df["target_churn"] = features_df["customer_id"].isin(targ_customers).apply(lambda x: 0 if x else 1)
    
    # Target 2: Customer Lifetime Value (CLV - total spend in next 90 days)
    df_targ_clean["total_spend"] = df_targ_clean["quantity"] * df_targ_clean["unit_price"]
    targ_spend = df_targ_clean.groupby("customer_id")["total_spend"].sum()
    features_df["target_clv"] = features_df["customer_id"].map(targ_spend).fillna(0.0)
    
    # Target 3: Next Purchase Category (First purchase stock code in the 90 days after cutoff)
    first_purchase = df_targ_clean.sort_values("invoice_date").groupby("customer_id")["stock_code"].first()
    features_df["target_next_category"] = features_df["customer_id"].map(first_purchase).fillna("NONE")
    
    # Drop intermediate columns
    cols_to_drop = ["frequency_last_90d", "frequency_prior_90d", "total_purchased_qty", "total_returned_qty", "discount_count"]
    features_df.drop(columns=cols_to_drop, inplace=True)
    
    if save_path:
        features_df.to_csv(save_path, index=False)
        print(f"Features and targets exported to {save_path}")
        
    return features_df
