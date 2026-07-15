import pandas as pd

#Part 1 Standardization 
def clean_text_base(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 1. Handle missing values
    df["description"] = df["description"].fillna("UNKNOWN ITEM")

    # 2. Case and outer spaces
    df["description"] = df["description"].astype(str).str.upper().str.strip()

    # 3. Inner multiple spaces
    df["description"] = df["description"].str.replace(
        r"\s+", " ", regex=True
    )
    return df


def create_description_lookup(df: pd.DataFrame) -> dict:
    df = df.copy()

    # 1. Count how often each stock_code + description pair occurs
    counts = (
        df.groupby(["stock_code", "description"])
        .size()
        .reset_index(name="count")
    )

    # 2. Sort so the most frequent description for each stock_code is at the top
    counts = counts.sort_values(
        by=["stock_code", "count"], ascending=[True, False]
    )

    # 3. Drop duplicates keeping the first row per stock_code (the most common one)
    best_descriptions = counts.drop_duplicates(subset=["stock_code"], keep="first")

    # 4. Turn it into our lookup dictionary
    lookup_dict = dict(
        zip(best_descriptions["stock_code"], best_descriptions["description"])
    )

    # 5. Apply manual corporate overrides
    known_typos = {
        "POST": "POSTAGE",
        "D": "DISCOUNT",
        "M": "MANUAL",
        "BANK CHARGES": "BANK CHARGES",
    }

    for code, clean_desc in known_typos.items():
        if code in lookup_dict:
            lookup_dict[code] = clean_desc

    return lookup_dict


def standardize_descriptions(
    df: pd.DataFrame, lookup_table: dict
) -> pd.DataFrame:
    df = df.copy()
    df["description"] = (
        df["stock_code"].map(lookup_table).fillna(df["description"])
    )
    return df


#Part 2 Validation
def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    expected_schema = {
        "invoice_no": "str",
        "stock_code": "str",
        "description": "str",
        "quantity": "int64",
        "invoice_date": "str",
        "unit_price": "float64",
        "customer_id": "float64",
        "country": "str",
    }

    for col, expected_type in expected_schema.items():
        if col not in df.columns:
            raise ValueError(f"❌ CRITICAL ERROR: Missing column: '{col}'")

        actual_type = str(df[col].dtype)

        if expected_type == "str":
            if "str" not in actual_type and "string" not in actual_type:
                raise TypeError(
                    f"❌ CRITICAL ERROR: Column '{col}' is type '{actual_type}', expected a string type!"
                )
        else:
            if expected_type not in actual_type:
                raise TypeError(
                    f"❌ CRITICAL ERROR: Column '{col}' is type '{actual_type}', expected '{expected_type}'!"
                )

    print("✅ Schema Validation Passed Successfully!")
    return df

def validate_null_rates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    max_null_thresholds = {
        "invoice_no": 0.0,
        "stock_code": 0.0,
        "customer_id": 0.25,
    }

    for col, max_allowed in max_null_thresholds.items():
        if col not in df.columns:
            raise ValueError(f"CRITICAL ERROR: Missing column: '{col}'")
            
        actual_null_rate = df[col].isna().mean()

        if actual_null_rate > max_allowed:
            raise ValueError(
                f" CRITICAL ERROR: '{col}' null rate is {actual_null_rate:.2%}. Max allowed is {max_allowed:.2%}"
            )

    print(" Null Rate Threshold Checks Passed!")
    return df


def validate_date_range(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # pd.to_datetime cleanly parses both object data formats and strict string sequences
    dates = pd.to_datetime(df["invoice_date"])

    min_date = pd.Timestamp("2009-12-01")
    max_date = pd.Timestamp("2011-12-31")

    actual_min = dates.min()
    actual_max = dates.max()

    if actual_min < min_date or actual_max > max_date:
        raise ValueError(
            f" CRITICAL ERROR: Date out of bounds! Found data from {actual_min} to {actual_max}. "
            f"Expected range: {min_date.date()} to {max_date.date()}"
        )

    print("x Date-Range Controls Passed!")
    return df