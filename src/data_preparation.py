import pandas as pd
import os
from typing import List

def load_application_data(path: str, is_train=True) -> pd.DataFrame:
    file_name = 'application_train.csv' if is_train else 'application_test.csv'
    full_path = os.path.join(path, file_name)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"🚨 Could not find {full_path}. Please check the file location.")
    
    return pd.read_csv(full_path)

def preprocess_bureau_data(path: str) -> pd.DataFrame:
    # Load files
    bureau = pd.read_csv(f"{path}/bureau.csv")
    bureau_balance = pd.read_csv(f"{path}/bureau_balance.csv")

    # ------------------------
    # Bureau Table Aggregation
    # ------------------------
    bureau_agg = bureau.groupby("SK_ID_CURR").agg({
        "AMT_CREDIT_SUM": ["mean", "sum"],
        "AMT_CREDIT_SUM_DEBT": ["mean", "sum"],
        "DAYS_CREDIT": "mean",
        "SK_ID_BUREAU": "count"
    })

    bureau_agg.columns = ["_".join(col).strip() for col in bureau_agg.columns.values]
    bureau_agg.reset_index(inplace=True)

    # Create new engineered features
    bureau_agg["DEBT_CREDIT_RATIO"] = bureau_agg["AMT_CREDIT_SUM_DEBT_sum"] / bureau_agg["AMT_CREDIT_SUM_sum"]
    bureau_agg["DEBT_CREDIT_RATIO"] = bureau_agg["DEBT_CREDIT_RATIO"].fillna(0)

    # ------------------------
    # Bureau Balance Aggregation
    # ------------------------
    # One-hot encode STATUS column
    bb_onehot = pd.get_dummies(bureau_balance, columns=["STATUS"])
    bb_agg = bb_onehot.groupby("SK_ID_BUREAU").mean().reset_index()

    # Merge with bureau to get SK_ID_CURR
    bureau_bb = bureau[["SK_ID_BUREAU", "SK_ID_CURR"]].merge(bb_agg, on="SK_ID_BUREAU", how="left")

    # Aggregate bureau_balance features at SK_ID_CURR level
    bb_final = bureau_bb.drop(columns=["SK_ID_BUREAU"]).groupby("SK_ID_CURR").mean().reset_index()

    # ------------------------
    # Merge bureau and bureau_balance features
    # ------------------------
    bureau_final = bureau_agg.merge(bb_final, on="SK_ID_CURR", how="left")

    return bureau_final


def preprocess_previous_applications(path: str) -> pd.DataFrame:
    prev = pd.read_csv(f"{path}/previous_application.csv")
    
    # Basic numeric aggregations per current client ID
    prev_agg = prev.groupby("SK_ID_CURR").agg({
        "AMT_APPLICATION": ["mean", "sum"],
        "AMT_CREDIT": ["mean", "sum"],
        "AMT_DOWN_PAYMENT": "mean",
        "AMT_ANNUITY": "mean",
        "CNT_PAYMENT": "mean",
        "DAYS_DECISION": "mean",
        "SK_ID_PREV": "count"  # total previous apps
    })

    # Flatten column names
    prev_agg.columns = ["PREV_" + "_".join(col).upper() for col in prev_agg.columns]
    prev_agg.reset_index(inplace=True)

    # Engineer credit-to-application ratio
    prev_agg["PREV_CREDIT_TO_APPLICATION_RATIO"] = (
        prev_agg["PREV_AMT_CREDIT_SUM"] / prev_agg["PREV_AMT_APPLICATION_SUM"]
    )

    prev_agg["PREV_CREDIT_TO_APPLICATION_RATIO"] = (
        prev_agg["PREV_CREDIT_TO_APPLICATION_RATIO"]
        .replace([float("inf"), -float("inf")], pd.NA)
        .astype("Float64")  # nullable float, preserves NaNs
    )

    return prev_agg

def preprocess_installments(path: str) -> pd.DataFrame:
    df = pd.read_csv(f"{path}/installments_payments.csv")

    # Time delay: positive = late, negative = early
    df["PAYMENT_DELAY"] = df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]

    # Ratio of paid to expected
    df["PAYMENT_RATIO"] = df["AMT_PAYMENT"] / df["AMT_INSTALMENT"]
    df["PAYMENT_RATIO"] = df["PAYMENT_RATIO"].replace([float("inf"), -float("inf")], None)

    # Flag for missed payments
    df["MISSED_PAYMENT"] = df["AMT_PAYMENT"].isna() | (df["AMT_PAYMENT"] == 0)

    # Group by client
    agg = df.groupby("SK_ID_CURR").agg({
        "PAYMENT_DELAY": ["mean", "max"],
        "PAYMENT_RATIO": "mean",
        "MISSED_PAYMENT": "sum",
        "AMT_PAYMENT": "sum"
    })

    # Clean column names
    agg.columns = ["INSTALL_" + "_".join(col).upper() for col in agg.columns]
    agg.reset_index(inplace=True)

    return agg

def merge_all_features(app_df: pd.DataFrame, feature_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge main application data with a list of engineered feature DataFrames.
    
    Parameters:
        app_df (pd.DataFrame): Main application_train or application_test dataframe.
        feature_dfs (List[pd.DataFrame]): List of feature tables to merge, each with SK_ID_CURR.
        
    Returns:
        pd.DataFrame: Enriched application dataframe with merged features.
    """
    for feature_df in feature_dfs:
        app_df = app_df.merge(feature_df, on="SK_ID_CURR", how="left")
    
    return app_df

def preprocess_final(df: pd.DataFrame, is_train=True) -> pd.DataFrame:
    """
    Final preprocessing of full dataset after feature merging.
    Handles missing values, encodes categoricals, etc.
    
    Parameters:
        df (pd.DataFrame): Full merged dataset
        is_train (bool): Whether this is training data
    
    Returns:
        pd.DataFrame: Cleaned dataset ready for modeling
    """
    # Fill numeric missing with median
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Label encode binary categoricals
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].nunique() == 2:
            df[col] = pd.factorize(df[col])[0]

    # Optional: drop columns with too many missing values, constant columns, etc.
    # Optional: Save feature metadata (dtype, unique count, etc.)

    return df

def prepare_dataset(path: str, is_train: bool = True) -> pd.DataFrame:
    app_df = load_application_data(path, is_train)
    
    # Feature aggregations
    bureau_df = preprocess_bureau_data(path)
    prev_app_df = preprocess_previous_applications(path)
    inst_df = preprocess_installments(path)

    # Merge all features into one
    full_df = merge_all_features(app_df, [bureau_df, prev_app_df, inst_df])
    
    # Final cleaning
    full_df = preprocess_final(full_df, is_train)

    return full_df
