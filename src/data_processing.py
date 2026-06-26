from typing import Literal

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


def get_corr_series(df: pd.DataFrame, col: str) -> pd.Series:
    return (
        df.corr(numeric_only=True)[col]
        .drop(col)
        .sort_values(key=abs, ascending=False)
    )


def get_df_filtered_outliers_by_irq_and_col(
    df: pd.DataFrame,
    col: str,
    # Mild outliers or Extreme outliers
    ol_type: Literal['mild', 'extreme'] = 'mild',
) -> pd.DataFrame:
    if ol_type == 'mild':
        multiplier = 1.5
    elif ol_type == 'extreme':
        multiplier = 3
    else:
        raise ValueError("'ol_type' need's to be 'mild' or 'extreme'")

    QUANTILE = 0.25
    q1 = df[col].quantile(QUANTILE)
    q3 = df[col].quantile(1 - QUANTILE)
    irq = q3 - q1

    return df[
        df[col].between(
            q1 - (multiplier * irq),
            q3 + (multiplier * irq)
        )
    ]


def get_df_filtered_outliers_by_irq(
    df: pd.DataFrame,
    # Mild outliers or Extreme outliers
    ol_type: Literal['mild', 'extreme'] = 'mild',
) -> pd.DataFrame:
    if ol_type == 'mild':
        multiplier = 1.5
    elif ol_type == 'extreme':
        multiplier = 3
    else:
        raise ValueError("'ol_type' need's to be 'mild' or 'extreme'")

    QUANTILE = 0.25
    q1 = df.quantile(QUANTILE)
    q3 = df.quantile(1 - QUANTILE)
    irq = q3 - q1

    lower = q1 - (multiplier * irq)
    upper = q3 + (multiplier * irq)
    mask = (df >= lower) & (df <= upper)
    return df[mask.all(axis=1)]


def get_df_filtered_outliers_by_quantile(df: pd.DataFrame, quantile=0.01) -> pd.DataFrame:
    lower = df.quantile(quantile)
    upper = df.quantile(1 - quantile)
    mask = (df >= lower) & (df <= upper)
    return df[mask.all(axis=1)]


def lof_outlier_detection(
    df_train: pd.DataFrame,
    n_neighbors: int = 20,
    use_pca: bool = True,
) -> pd.DataFrame:
    contamination = 0.05
    pca_variance = 0.95

    df = df_train.drop(columns=["Id", "SalePrice"], errors="ignore")

    # one-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    # missing values
    df = df.fillna(df.median(numeric_only=True))

    # scaling
    X = StandardScaler().fit_transform(df)

    # optional PCA (recomendado para House Prices)
    if use_pca:
        X = PCA(n_components=pca_variance).fit_transform(X)

    # LOF
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination
    )

    labels = lof.fit_predict(X)
    scores = lof.negative_outlier_factor_

    result = df_train.copy()
    result["is_outlier"] = labels == -1
    result["lof_score"] = scores

    return result.sort_values("lof_score")
