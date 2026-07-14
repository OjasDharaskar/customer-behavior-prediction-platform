import numpy as np
import pandas as pd

#Renaming Columns
def rename_column(df:pd.DataFrame) ->pd.DataFrame:
    column_mapping = {
    "Invoice": "invoice_no",
    "StockCode": "stock_code",
    "Description":"description",
    "Quantity":"quantity",
    "InvoiceDate":"invoice_date",
    "Price":"unit_price",
    "Customer ID":"customer_id",
    "Country":"country"
    }
    df=df.rename(columns=column_mapping)
    return df

#Removing Duplicates

def remove_duplicates(df:pd.DataFrame)->pd.DataFrame:
    composite_key = [
    "invoice_no",
    "stock_code",
    "quantity",
    "unit_price",
    "invoice_date",
    "customer_id"
    ]
    existing_keys=[cols for cols in composite_key if cols in df.columns]
    return df.drop_duplicates(subset=existing_keys,keep="first")

def filter_valid_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_returns = df[(df["unit_price"] < 0) | (df["quantity"] < 0)].copy()
    df_returns = df_returns[df_returns["unit_price"] >= 0]

    df_clean = df[(df["unit_price"] >= 0) & (df["quantity"] >= 0)].copy()

    return df_clean, df_returns


def flag_order_anomalies(df: pd.DataFrame, qnt_upper_bound: float) -> pd.DataFrame:
    """Computes invoice sizes and item median prices to flag bulk orders or pricing errors."""
    df = df.copy()
    invoice_items = (
        df.groupby("invoice_no").size().rename("items_in_invoice")
    )
    df = df.merge(invoice_items, on="invoice_no", how="left")
    median_prices = (
        df.groupby("stock_code")["unit_price"].median().rename("median_price")
    )
    df = df.merge(median_prices, on="stock_code", how="left")

    df["flag"] = "Normal"

    bulk_mask = (df["quantity"] > qnt_upper_bound) & (
        df["items_in_invoice"] >= 10
    )
    price_error_mask = df["unit_price"] > (10 * df["median_price"])

    df.loc[bulk_mask, "flag"] = "Likely Bulk Order"
    df.loc[price_error_mask, "flag"] = "Possible Pricing Error"

    return df    