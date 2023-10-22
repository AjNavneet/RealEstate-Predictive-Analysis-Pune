import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

import configparser
config = configparser.RawConfigParser()
config.read('..\\input\\config.ini')
OUTPUT_DIR = config.get('DATA', 'output_dir')

# Function to read the data file
def read_data(file_path, **kwargs):
    raw_data = pd.read_excel(file_path, **kwargs)
    return raw_data

# Function to read the data file
def read_data_csv(file_path, **kwargs):
    raw_data = pd.read_csv(file_path, **kwargs)
    return raw_data

# Function to drop columns from data
def drop_col(df, col_list):
    for col in col_list:
        if col not in df.columns:
            raise ValueError(f"Column does not exist in the dataframe")
        else:
            df = df.drop(col, axis=1)
    return df

# Function to scale the data
def data_scale(data, df_col):
    # Standard scaling for surface
    sc = StandardScaler(with_std=True, with_mean=True)
    data[df_col] = sc.fit_transform(data[[df_col]])
    return data

# Function to plot feature importances
def feature_imp_plot(model_cv):
    # Find the best features
    rfr = model_cv.best_estimator_
    df_imp = pd.DataFrame(zip(rfr.feature_names_in_, rfr.feature_importances_))
    df_imp.columns = ["feature", "importance"]
    df_imp = df_imp.sort_values("importance", ascending=False).reset_index(drop=True)
    sns.barplot(data=df_imp, x="importance", y="feature")
    plt.savefig(OUTPUT_DIR + "feature_imp_plot.png")

# Price min-max normalization to be used with tanh activation
def y_scaled_val(y):
    R = np.max(y) - np.min(y)
    y0 = np.min(y)
    y_scaled = 2 * (y - y0) / R - 1
    return y_scaled

# y2price function
def y2price(y, R):
    """
    Convert the scaled price to the normal price
    Args:
    y: list, series, array
        scaled price
    R: float
        scale factor
    """
    return y # R * (y + 1) / 2. + y0
