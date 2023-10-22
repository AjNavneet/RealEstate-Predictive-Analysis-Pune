from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Function for performing label encoding for the categorical variables
def encode_categorical_variables(df, cat_vars):
    """
    Encode categorical variables with LabelEncoder.
    Parameters:
        df (DataFrame): Input DataFrame.
        cat_vars (list): List of categorical variable column names.
    Returns:
        DataFrame: Encoded DataFrame.
    """
    laben = LabelEncoder()
    for col in cat_vars:
        df[col] = laben.fit_transform(df[col])
    return df

# Function for feature engineering based on sub-area count
def fea_eng_sa(df_count, df_col, df, n):
    """
    Feature engineering based on sub-area count.
    Parameters:
        df_count (DataFrame): DataFrame with sub-area counts.
        df_col (str): Name of the sub-area column.
        df (DataFrame): Input DataFrame.
        n (int): Minimum count threshold.
    Returns:
        DataFrame: DataFrame with sub-areas meeting the count threshold.
    """
    sa_sel_col = df_count.loc[df_count["count"] > n, df_col].to_list()
    df[df_col] = df[df_col].where(df[df_col].isin(sa_sel_col), "other")
    return df

# Function to perform one-hot encoding
def onehot_end(df, col_name):
    """
    Perform one-hot encoding on a specified column.
    Parameters:
        df (DataFrame): Input DataFrame.
        col_name (str): Name of the column to one-hot encode.
    Returns:
        array: One-hot encoded features.
    """
    hoten = OneHotEncoder(sparse=False)
    X_dummy = hoten.fit_transform(df[[col_name]])
    return X_dummy
