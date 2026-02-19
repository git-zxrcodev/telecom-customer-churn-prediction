import pandas as pd


# Columns dropped after feature engineering -- these are replaced by engineered features
# or found non-significant via chi-squared test in notebook 02.
_COLS_TO_DROP = ['gender', 'phone_service', 'contract', 'total_charges']


def feature_engineering(df: pd.DataFrame, drop_replaced: bool = False) -> pd.DataFrame:
    """
    Perform feature engineering on the telecom customer churn dataset.

    Steps:
    - Validate tenure >= 1 (tenure == 0 rows should be removed upstream).
    - Create 'high_risk_tenure' based on 'tenure'.
    - Create 'contract_stability' based on 'contract' (ordinal encoding).
    - Create 'fiber_no_support' based on 'internet_service' and 'tech_support'.
    - Create 'high_risk_new_monthly' based on 'tenure' and 'contract'.
    - Create 'manual_payment_early' based on 'payment_method' and 'tenure'.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with post-wrangling columns.
    drop_replaced : bool, default False
        If True, drop the columns that are replaced by engineered features
        ('gender', 'phone_service', 'contract', 'total_charges').

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    # --- Validation ----------------------------------------------------------
    assert (df['tenure'] >= 1).all(), (
        "tenure contains values < 1. Rows with tenure == 0 should be dropped "
        "during preprocessing (they have NaN total_charges)."
    )

    # --- Tenure risk category ------------------------------------------------
    df.loc[:, 'high_risk_tenure'] = pd.cut(
        df['tenure'],
        bins=[0, 4, 12, 100],
        labels=['high_risk_category', 'medium_risk_category', 'low_risk_category']
    )

    # --- Contract stability (ordinal) ----------------------------------------
    contract_stability = {'Month-to-month': 1, 'One year': 2, 'Two year': 3}
    df['contract_stability'] = df['contract'].map(contract_stability)
    assert df['contract_stability'].notna().all(), (
        f"Unknown contract type(s) found: "
        f"{df.loc[df['contract_stability'].isna(), 'contract'].unique().tolist()}"
    )

    # --- Binary interaction features -----------------------------------------
    df['fiber_no_support'] = (
        (df['internet_service'] == 'Fiber optic') &
        (df['tech_support'] == 'No')
    ).astype(int)

    df['manual_payment_early'] = (
        (df['payment_method'].isin(['Electronic check', 'Mailed check'])) &
        (df['tenure'] <= 6)
    ).astype(int)

    df['high_risk_new_monthly'] = (
        (df['tenure'] <= 6) &
        (df['contract'] == 'Month-to-month')
    ).astype(int)

    new_boolean_features = [
        'fiber_no_support',
        'manual_payment_early',
        'high_risk_new_monthly'
    ]

    df[new_boolean_features] = df[new_boolean_features].astype(bool)

    # --- Optionally drop columns replaced by engineered features -------------
    if drop_replaced:
        cols_present = [c for c in _COLS_TO_DROP if c in df.columns]
        df = df.drop(columns=cols_present)

    return df