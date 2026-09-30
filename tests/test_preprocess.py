"""
test_preprocess.py, Unit tests for the SmartLend preprocessing pipeline.
"""

import pandas as pd
import numpy as np
import pytest
import sys
import os

# Allow the test runner to find src/ without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocess import (
    impute_missing_values,
    validate_columns,
    remove_outliers,
    EXPECTED_COLUMNS,
    REQUIRED_NON_NULL,
)


def make_minimal_df():
    """Create a small DataFrame with the expected schema for testing."""
    data = {col: [1.0, 2.0, None] for col in EXPECTED_COLUMNS}
    df = pd.DataFrame(data)
    # Make age and SeriousDlqin2yrs realistic
    df["age"] = [35, 0, 55]
    df["SeriousDlqin2yrs"] = [0, 1, 0]
    df["RevolvingUtilizationOfUnsecuredLines"] = [0.5, 1.5, 0.3]
    return df


def test_imputed_columns_have_no_nulls():
    """After imputation, MonthlyIncome and NumberOfDependents must have no missing values."""
    df = make_minimal_df()
    # Introduce explicit nulls in the two target columns
    df.loc[0, "MonthlyIncome"] = np.nan
    df.loc[1, "NumberOfDependents"] = np.nan

    result = impute_missing_values(df)

    # Check that the two columns have no nulls
    assert result["MonthlyIncome"].isnull().sum() == 0, (
        "MonthlyIncome still contains null values after imputation"
    )
    assert result["NumberOfDependents"].isnull().sum() == 0, (
        "NumberOfDependents still contains null values after imputation"
    )


def test_processed_data_has_expected_columns():
    """The output of impute_missing_values must retain all expected columns."""
    df = make_minimal_df()
    result = impute_missing_values(df)

    for col in EXPECTED_COLUMNS:
        assert col in result.columns, f"Expected column '{col}' missing after preprocessing"