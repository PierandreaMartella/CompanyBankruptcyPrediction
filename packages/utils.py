import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def standard_OHE(df):
    """
    Trasforma le variabili binarie in 0 e 1, e one-hot encode le variabili categoriche in un DataFrame.
    """
    # Find Categorical Columns
    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Find Binary Columns
    binary_columns = [col for col in df.columns if df[col].nunique() == 2]

    # Convert binary columns to 0 and 1
    for col in binary_columns:
        unique_values = df[col].unique()
        df[col] = df[col].apply(lambda x: 0 if x == unique_values[0] else 1)

    # Apply One-Hot Encoding
    encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoded = encoder.fit_transform(df[categorical_columns])

    one_hot_df = pd.DataFrame(
        one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns)
    )

    # Join
    df_encoded = df.join(one_hot_df)

    # Remove unnecessary vars
    df_encoded = df_encoded.drop(categorical_columns, axis=1)

    return df_encoded

def standard_OHE(df, drop_first=False):
    """
    Trasforma le variabili binarie in 0 e 1, e one-hot encode le variabili categoriche in un DataFrame.
    """
    # Find Categorical Columns
    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Find Binary Columns
    binary_columns = [col for col in df.columns if df[col].nunique() == 2]

    # Convert binary columns to 0 and 1
    for col in binary_columns:
        unique_values = df[col].unique()
        df[col] = df[col].apply(lambda x: 0 if x == unique_values[0] else 1)

    # Apply One-Hot Encoding
    encoder = OneHotEncoder(sparse_output=False)
    if drop_first:
        encoder = OneHotEncoder(drop="first", sparse_output=False)
    one_hot_encoded = encoder.fit_transform(df[categorical_columns])

    one_hot_df = pd.DataFrame(
        one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns)
    )

    # Join
    df_encoded = df.join(one_hot_df)

    # Remove unnecessary vars
    df_encoded = df_encoded.drop(categorical_columns, axis=1)

    return df_encoded
