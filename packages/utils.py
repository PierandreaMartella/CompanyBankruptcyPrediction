import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def standard_OHE(df):
    """
    One-hot encodes the categorical columns in a DataFrame.
    """
    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoded = encoder.fit_transform(df[categorical_columns])

    one_hot_df = pd.DataFrame(
        one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns)
    )

    # Join the one-hot encoded columns to the original
    df_encoded = df.join(one_hot_df)

    # Drop the original categorical columns (redundant information)
    df_encoded = df_encoded.drop(categorical_columns, axis=1)
    return df_encoded
