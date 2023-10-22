import numpy as np

# Function to perform feature selection based on correlations
def feat_sel(data, corr_cols_list, target, col_name):
    """
    Feature selection based on correlations with the target variable.
    Parameters:
        data (DataFrame): Input DataFrame.
        corr_cols_list (list): List of column names to calculate correlations.
        target (str): Name of the target variable.
        col_name (list): List of sub-area columns.
    Returns:
        list: List of correlations between columns and the target variable.
    """
    # Remove the target variable from the list of columns to calculate correlations
    corr_cols_list.remove(target)
    corr_cols_list.extend(col_name)
    
    corr_list = []  # To store correlations with the target variable
    for col in corr_cols_list:
        corr_list.append(round(data[target].corr(data[col]), 2))
    
    return corr_list

# Function for feature engineering based on sub-area counts and mean price
def feature_sa(df, df_col, target, features):
    """
    Feature engineering based on sub-area counts and mean price.
    Parameters:
        df (DataFrame): Input DataFrame.
        df_col (str): Name of the sub-area column.
        target (str): Name of the target variable.
        features (list): List of feature columns.
    Returns:
        list: List of tuples with sub-area name, mean price, and count.
    """
    sa_feature_list = [sa for sa in features if "sa" in sa]
    lst = []
    for col in sa_feature_list:
        sa_trigger = df[col] == 1
        sa = df.loc[sa_trigger, df_col].to_list()[0]
        x = df.loc[sa_trigger, target]
        lst.append((sa, np.mean(x), df[col].sum()))
    
    return lst
