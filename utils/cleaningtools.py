"""Helper module for data cleaning and preprocessing"""

from tabulate import tabulate
import numpy as np
import pandas as pd
import re


def overview_data(df: pd.DataFrame, display_frst_value=False) -> None:
    """
    Overview data from a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to analyze.
        display_frst_value (bool): Whether to display the first value of each column.

    Returns:
        None. Prints the analysis results.
    """
    print(f"The DataFrame has {df.shape[0]} rows and {df.shape[1]} columns.")

    missing_values = df.isnull().sum().reset_index()
    missing_values.columns = ["Column", "NA Values"]
    missing_values["%"] = np.round((missing_values["NA Values"] / df.shape[0]) * 100, 1)

    data_types = df.dtypes.reset_index()
    data_types.columns = ["Column", "Data Type"]
    data_types.drop("Column", axis=1, inplace=True)

    unique_values = df.nunique().reset_index()
    unique_values.columns = ["Column", "Unique"]
    unique_values.drop("Column", axis=1, inplace=True)

    if display_frst_value:
        first_value = df.apply(
            lambda x: x.dropna().iloc[0] if not x.dropna().empty else np.nan
        )
        first_value = first_value.reset_index()
        first_value.columns = ["Column", "First Value"]
        first_value["First Value"] = first_value["First Value"].astype(str)
        first_value["First Value"] = np.where(
            first_value["First Value"].str.len() > 10,
            first_value["First Value"].str[:10] + "...",
            first_value["First Value"],
        )
        first_value.drop("Column", axis=1, inplace=True)
    else:
        first_value = pd.DataFrame(columns=None)

    result_table = pd.concat(
        [missing_values, data_types, unique_values, first_value], axis=1
    )

    print(tabulate(result_table, headers="keys", tablefmt="psql"))

    return None


def remove_missing_features(
    df: pd.DataFrame, threshold=0.5, feat_not_to_remove=[]
) -> pd.DataFrame:
    """
    Remove features with missing values based on the given threshold.

    Parameters:
    - df: pandas DataFrame
        Input DataFrame.
    - threshold: float, optional (default=0.5)
        Threshold for missing value percentage. Features with missing values
        percentage greater than this threshold will be removed.
    - feat_not_to_remove: list, optional (default=[])

    Returns:
    - df_cleaned: pandas DataFrame
        DataFrame with missing features removed.
    - removed_features: list
        List of features that were removed.
    """
    missing_percentages = df.isnull().mean()
    features_to_remove = missing_percentages[
        missing_percentages > threshold
    ].index.tolist()
    features_to_remove = [
        feat for feat in features_to_remove if feat not in feat_not_to_remove
    ]
    df_cleaned = df.drop(columns=features_to_remove)
    return df_cleaned, features_to_remove


def convert_types(df: pd.DataFrame, print_info=False) -> pd.DataFrame:
    original_memory = df.memory_usage().sum()

    for c in df:
        if ("SK_ID" in c) or ("sk_id" in c):
            df[c] = df[c].fillna(0).astype(np.int32)

        elif (df[c].dtype == "object") and (df[c].nunique() < df.shape[0]):
            df[c] = df[c].astype("category")

        elif list(df[c].unique()) == [1, 0] or list(df[c].unique()) == [0, 1]:
            df[c] = df[c].astype("category")

        elif df[c].dtype == float:
            df[c] = df[c].astype(np.float32)

        elif df[c].dtype == int:
            df[c] = df[c].astype(np.int32)

    new_memory = df.memory_usage().sum()

    if print_info:
        print(f"Original Memory Usage: {round(original_memory / 1e9, 2)} gb.")
        print(f"New Memory Usage: {round(new_memory / 1e9, 2)} gb.")

    return df


def remove_special_characters(input_string: str):
    """Remove special characters from the string and replace them with '_'. Lowercase the string.
    Returns the modified string."""
    pattern = r"[^a-zA-Z0-9_]"
    result_string = re.sub(pattern, "_", input_string)
    result_string = result_string.lower()
    return result_string


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Removes special characters from column names and make the names lowercase."""
    rename_mapping = {}
    for c in df.columns:
        r = remove_special_characters(c)
        rename_mapping[c] = r
    return df.rename(columns=rename_mapping)
