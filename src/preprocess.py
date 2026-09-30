"""
preprocess.py, SmartLend data preprocessing pipeline.

Loads the raw Give Me Some Credit dataset, applies cleaning steps,
and saves a processed version ready for model training.
"""

import pandas as pd
import numpy as np
from pathlib import Path


EXPECTED_COLUMNS = [
    "SeriousDlqin2yrs",
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

REQUIRED_NON_NULL = [
    "SeriousDlqin2yrs",
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "DebtRatio",
]


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load the raw CSV dataset. Drops the unnamed index column if present."""
    df = pd.read_csv(filepath)
    # The Kaggle dataset includes an unnamed index column
    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    return df


def validate_columns(df: pd.DataFrame) -> None:
    """Raise ValueError if expected columns are missing."""
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values using median imputation.

    MonthlyIncome and NumberOfDependents are the two columns with
    missing data in the Give Me Some Credit dataset.
    """
    df = df.copy()
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(
        df["MonthlyIncome"].median()
    )
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(
        df["NumberOfDependents"].median()
    )
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply basic outlier filtering.
    RevolvingUtilizationOfUnsecuredLines > 1.0 represents values that
    exceed 100% credit utilisation.
    Age values of 0 or below are data entry errors.
    """
    df = df.copy()

    df = df[
        (df["RevolvingUtilizationOfUnsecuredLines"] <= 1.0)
        & (df["age"] > 0)
    ]

    return df


def preprocess(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Full preprocessing pipeline: load, validate, impute, filter, save.

    Returns the processed DataFrame for inspection or testing.
    """
    df = load_raw_data(input_path)
    validate_columns(df)
    df = impute_missing_values(df)
    df = remove_outliers(df)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    preprocess(
        input_path="data/raw/cs-training.csv",
        output_path="data/processed/cs-processed.csv",
    )