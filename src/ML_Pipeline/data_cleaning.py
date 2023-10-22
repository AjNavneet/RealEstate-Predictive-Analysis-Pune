# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Function to rename a column
def rename_col(data, column_name, new_column_name):
    data.rename(columns={column_name: new_column_name}, inplace=True)
    return data

# Function to drop a specific row from a column
def drop_val(data, column, row):
    data = data[data[column] != row]
    return data

# Function for data cleaning of 'Property Area in Sq. Ft'
def splitSum(e, flag=True):
    """
    Gives the total number of bedrooms or property area.
    """
    try:
        e = str(e).lower()
        e = re.sub(r"[,;@#?!&$+]+\ *", " ", e)
        e = re.sub(r"[a-z]+", " ", e)
        e = re.sub(r"\s\s", "", e)

        s2list = e.strip().split()
        sumList = sum(float(e) for e in s2list)

        e_norm = sumList if flag else sumList / len(s2list)
        return e_norm
    except:
        return np.nan

# Function to normalize data
def normaliseProps(df):
    """
    Extracts the number of rooms from 'Property Type' columns and mean values for "Property Area in Sq. Ft."
    """
    data = df.copy()
    data["Property Type"] = data["Property Type"].apply(splitSum)
    data["Property Area in Sq. Ft."] = data["Property Area in Sq. Ft."].apply(lambda x: splitSum(x, False))
    return data

# Function to compute the upper/lower fence for outliers
def computeUpperFence(df_col, up=True):
    """
    Computes the upper/lower fence for a given column.
    """
    iqr = df_col.quantile(0.75) - df_col.quantile(0.25)
    if up:
        return df_col.quantile(0.75) + iqr * 1.5
    return df_col.quantile(0.25) - iqr * 1.5

# Function to compute the rate of non-NaNs for each column
def compute_fill_rate(df):
    """
    Computing the rate of non-NaNs for each column.
    """
    fr = pd.DataFrame(1 - df.isnull().sum().values.reshape(1, -1) / df.shape[0], columns=df.columns)
    return fr

# Function to plot the fill rate for columns
def plot_fill_rate(df):
    """
    Plot the fill rate.
    """
    fill_rate = compute_fill_rate(df)
    fig, ax = plt.subplots(figsize=(18, 18))
    sns.barplot(data=fill_rate, orient="h")
    ax.set_title("Fill rate for columns", fontsize=28)
    ax.set(xlim=(0, 1.))
    plt.show()

# Function to drop columns or rows with fill rates below a specified threshold
def drop_empty_axis(df, minFillRate, axis=1):
    """
    Drops axes that do not meet the minimum non-NaN rate.
    """
    i = 0 if axis == 1 else 1
    return df.dropna(axis=axis, thresh=int(df.shape[i] * minFillRate))

# Function to print unique values in columns
def print_uniques(cols, df):
    for col in cols:
        list_unique = df[col].unique()
        list_unique.sort()
        print(col, ":\n", list_unique)
        print("Number of unique categories:", len(list_unique))
        print("--------------------")

# Function to regularize categorical variables
def reg_catvar(df, cols):
    for col in cols:
        df[col] = df[col].apply(lambda x: x.strip().lower())
    return cols
