# Importing required packages
import pandas as pd
from sklearn.model_selection import train_test_split

# Import custom utility functions and modules from 'ML_Pipeline' package
from ML_Pipeline.utils import *
from ML_Pipeline.data_cleaning import *
from ML_Pipeline.fea_engg import encode_categorical_variables, fea_eng_sa, onehot_end
from ML_Pipeline.feat_sel import feat_sel, feature_sa
from ML_Pipeline.models import final_model
from ML_Pipeline.mlp_model import *
from ML_Pipeline.mlp_model_evaluation import *

# Import the 'configparser' module for configuration file handling
import configparser

# Read configuration file
config = configparser.RawConfigParser()
config.read('..\\input\\config.ini')
DATA_DIR = config.get('DATA', 'data_dir')
OUTPUT_DIR = config.get('DATA', 'output_dir')

# Read the initial dataset
dfr = read_data(DATA_DIR)

# DATA CLEANING
# Rename a column
dfr = rename_col(dfr, "Propert Type", "Property Type")

# Drop rows with 'shop' in the 'Property Type' column
dfr = drop_val(dfr, "Property Type", "shop")

# Normalize 'Property Type' and 'Property Area in Sq. Ft.'
df_norm = normaliseProps(dfr)

# Check for outliers in 'Property Type' and remove them
x_prt = df_norm['Property Type']
prt_up_lim = computeUpperFence(x_prt)
df_norm[x_prt > prt_up_lim]
df_norm.drop(index=86, inplace=True)
df_norm.drop(index=df_norm[df_norm["Property Type"] == 7].index, inplace=True)

# Data selection - Choose one of two target variables, 'Price in lakhs' or 'Price in millions'
df_norm["Price in lakhs"] = df_norm["Price in lakhs"].apply(lambda x: pd.to_numeric(x, errors='coerce'))
df_norm = drop_col(df_norm, ["Price in lakhs"])

# Dealing with missing values
compute_fill_rate(df_norm)

# Sort and display columns related to 'Sub-Area', 'TownShip Name/ Society Name', and 'Total TownShip Area in Acres'
df_norm[["Sub-Area", "TownShip Name/ Society Name", "Total TownShip Area in Acres"]].sort_values("Sub-Area").reset_index(drop=True)

# Drop columns with fill rate less than 50%
df_norm = drop_empty_axis(df_norm, minFillRate=.5)

# Regularize the categorical columns
binary_cols = df_norm.iloc[:, -7:].columns.to list()
df_norm = df_norm[df_norm["Price in Millions"] < 80]
binary_cols = reg_catvar(df_norm, binary_cols)

obj_cols = df_norm.select_dtypes(include="object").columns.to list()
multiCat_cols = list(set(obj_cols) ^ set(binary_cols))
multiCat_cols = reg_catvar(df_norm, multiCat_cols)

# Drop the 'Location' column
df_norm = drop_col(df_norm, ["Location"])

# Rename the columns
df_norm.columns = ["index", "sub_area", "n_bhk", "surface", "price",
                   "company_name", "township", "club_house", "school", "hospital",
                   "mall", "park", "pool", "gym"]

# Save the cleaned dataframe to a CSV file
df_norm.to_csv("../input/resd_clean.csv", index=False)

# DATA ANALYSIS
# Read the cleaned dataset
df = read_data_csv('../input/resd_clean.csv')

# Drop unnecessary columns
df = drop_col(df, ["index", "company_name", "township"])

# Remove duplicate rows
df = df.drop_duplicates()

# Convert binary columns
binary_cols = df.iloc[:, 4:].columns.to_list()
df = encode_categorical_variables(df, binary_cols)

# Calculate the contribution of different sub-areas in the dataset
df_sa_count = df.groupby("sub_area")["price"].count().reset_index()\
                .rename(columns={"price": "count"})\
                .sort values("count", ascending=False)\
                .reset_index(drop=True)
df_sa_count["sa_contribution"] = df_sa_count["count"] / len(df)

# Perform feature engineering on 'sub_area'
df = fea_eng_sa(df_sa_count, "sub_area", df, 7)

# Create one-hot encoded columns for 'sub_area'
X_dummy = onehot_end(df, "sub_area")
X_dummy = X_dummy.astype("int64")

# Rename the dummy columns
sa_cols_name = ["sa" + str(i + 1) for i in range(X_dummy.shape[1])]
df.loc[:, sa_cols_name] = X_dummy

# Display the relationship between 'sub_area' and dummy columns
df[["sub_area"] + sa_cols_name].drop_duplicates()\
            .sort_values("sub_area").reset_index(drop=True)

# Select only object datatype columns
data = df.select_dtypes(exclude="object")

# Extract float columns from the data
float_cols = data.select_dtypes(include="float").columns.to_list()

# Calculate correlations between columns and 'price'
corr_cols_list = float_cols + binary_cols
corr_list = feat_sel(data, corr_cols_list, "price", sa_cols_name)

# Create a DataFrame with column names and their correlations
df_corr = pd.DataFrame(data=zip(corr_cols_list, corr_list),
                 columns=["col_name", "corr"])\
            .sort_values("corr", ascending=False)\
            .reset_index(drop=True)

# Select features with a correlation magnitude greater than 0.1
features = df_corr.loc[abs(df_corr["corr"]) > 0.1, "col_name"].to_list()

# Perform feature selection for 'sub_area' and 'price'
lst = feature_sa(df, "sub_area", "price", features)

# Data scaling
sel_data = data[features + ["price"]].copy()
sel_data = data_scale(sel_data, "surface")

# Save the processed data to a CSV file
sel_data.to_csv("../input/resd_features.csv", index=False)

# MODELS
# Read the final CSV data
data = read_data_csv("../input/resd_features.csv")
data = data.sort_values("surface").reset_index(drop=True)

# Select feature matrix (X) and target vector (y)
X = data.iloc[:, :-1]
y = data["price"]

# Split the data into training and testing sets
rs = 118
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=rs)

# Regression models - Model Building
model_reg = final_model('random', X, y, rs, X_train, X_test, y_train, y_test)
print("Regression Model Executed")

# MLP with TensorFlow
rs = 13
y_scaled = y_scaled_val(y)
X_train, X_test, y_train, y_test = train_test_split(X.values, y_scaled.values,
                                                    test_size=.3,
                                                    random_state=rs)

# Build and evaluate an MLP model using TensorFlow
mlp_tensorflow_model = mlp_tf_model(X_train, X_test, y_train, y_test)
mlp_tensorflow_model.summary()

# Plot the training history, residuals, and predictions for the MLP model
plot_model_history(mlp_tensorflow_model.history)
plotResidueMLP(mlp_tensorflow_model, X, y_scaled, rs=rs)
plot_real_pred(mlp_tensorflow_model, X, y_scaled, rs)
print("MLP Model Executed")
