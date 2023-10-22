import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from ML_Pipeline.model_evaluation import gridSearchReport, plotResidue
from ML_Pipeline.xgboost import xgboost_model
from ML_Pipeline.utils import feature_imp_plot
import pickle

import configparser
config = configparser.RawConfigParser()
config.read('..\\input\\config.ini')
DATA_DIR = config.get('DATA', 'data_dir')
OUTPUT_DIR = config.get('DATA', 'output_dir')

# Function to build a linear regression model
def reg_model(X_train, X_test, y_train, y_test, X, y, rs):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    score = r2_score(y_train, lr.predict(X_train)), r2_score(y_test, lr.predict(X_test)) # Check for the R-squared score on the train and test data
    return score

# Function to build a ridge regression model
def ridge_model(X, y, rs):
    alphas = np.logspace(-3, 3, 100) # Define the parameters for the ridge regression model and perform grid search
    pg = {"alpha": alphas}
    ridge = Ridge() # Define a ridge model
    ridg_cv = gridSearchReport(ridge, X, y, pg, rs=rs)
    plotResidue(ridg_cv, X, y, rs) # Plot the residuals
    return ridg_cv

# Function to build a lasso regression model
def lasso_model(X, y, rs):
    lasso = Lasso()
    alphas = np.logspace(-3, 3, 100)
    pg = {"alpha": alphas} # Define the parameters and perform grid search for lasso regression
    lasso_cv = gridSearchReport(lasso, X, y, pg, rs=rs)
    plotResidue(lasso_cv, X, y, rs) # Plot the residuals
    return lasso_cv

# Function to build an elastic net regression model
def elasticnet_model(X, y, rs):
    l1_ratio = np.random.rand(20)
    elastic = ElasticNet() # Define an elastic net model
    pg = {"alpha": np.linspace(0.1, 1, 5), "l1_ratio": l1_ratio}
    elastic_cv = gridSearchReport(elastic, X, y, pg, rs=rs)
    plotResidue(elastic_cv, X, y, rs) # Plot the residuals
    return elastic_cv

# Define an SVR model
def svr_model(X, y, rs):
    svr = SVR() # Define an SVR model
    pg = {
        "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
        "gamma": ['scale', 'auto'], # Kernel coefficient
        "C": np.logspace(-3, 3, 10), # Penalty parameter
        "epsilon": np.linspace(.1, 1., 10) # The decision boundary
    }
    svr_cv = gridSearchReport(svr, X, y, pg, cv=10, rs=rs)
    plotResidue(svr_cv, X, y, rs) # Plot the residuals
    return svr_cv

# Random Forest model
def rand_fr(X_train, X, y, rs):
    rfr = RandomForestRegressor(random_state=10) # Define a random forest regressor
    pg = {
        "n_estimators": [10, 20, 30, 50], # Define parameters
        "criterion": ["squared_error", "absolute_error", "poisson"],
        "max_depth": [2, 3, 4],
        "min_samples_split": range(2, 10),
        "min_samples_leaf": [2, 3],
        "max_features": range(4, X_train.shape[1] + 1)
    }
    rfr_cv = gridSearchReport(rfr, X, y, pg, cv=5, rs=rs)
    best_fea = feature_imp_plot(rfr_cv)
    plotResidue(rfr_cv, X, y, rs) # Plot the residuals
    return rfr_cv

# K-Nearest Neighbors (KNN) model
def knn_model(X_train, X_test, y_train, y_test):
    knn_cv = KNeighborsRegressor(n_neighbors=20, weights="uniform") # Define a KNN model
    knn_cv.fit(X_train, y_train) # Fit the train-test data
    y_pred_knn = knn_cv.predict(X_test) # Predict on the test set
    r2 = r2_score(y_train, knn_cv.predict(X_train)), r2_score(y_test, y_pred_knn) # Calculate R-squared score for train and test data
    mse_score = mean_squared_error(y_train, knn_cv.predict(X_train)), mean_squared_error(y_test, y_pred_knn) # Calculate mean squared error (MSE) for train and test data
    cross_val_list = cross_val_score(knn_cv, X_train, y_train, scoring="neg_mean_squared_error", cv=10) # Cross-validation score
    score_val_knn = -np.mean(cross_val_list)
    print("r2 score", r2,
          "MSE score", mse_score,
          "val _score", score_val_knn)

# Function to choose and build the final model
def final_model(model_type, X, y, rs, X_train, X_test, y_train, y_test):
    if model_type == 'linear':
        model = reg_model(X_train, X_test, y_train, y_test, X, y, rs)
        print("LINEAR REGRESSION MODEL: ")
        pickle.dump(model, open(OUTPUT_DIR + 'reg_model.pkl', 'wb'))
    elif model_type == 'ridge':
        model = ridge_model(X, y, rs)
        print("RIDGE REGRESSION MODEL: ")
        pickle.dump(model, open(OUTPUT_DIR + 'ridge_model.pkl', 'wb'))
    elif model_type == 'lasso':
        model = lasso_model(X, y, rs)
        print("LASSO REGRESSION MODEL: ")
        pickle.dump(model, open(OUTPUT_DIR + 'lasso_model.pkl', 'wb'))
    elif model_type == 'elasticnet':
        model = elasticnet_model(X, y, rs)
        print("ELASTICNET MODEL: ")
        pickle.dump(model, open(OUTPUT_DIR + 'elastic_model.pkl', 'wb'))
    elif model_type == 'svr':
        model = svr_model(X, y, rs)
        print("SVR MODEL: ")
        pickle.dump(model, open(OUTPUT_DIR + 'svr_model.pkl', 'wb'))
    elif model_type == 'random':
        model = rand_fr(X_train, X, y, rs)
        print("RANDOM MODEL: ")
        pickle.dump(model, open(OUTPUT_DIR + 'randomforest_model.pkl', 'wb'))
    elif model_type == 'knn':
        model = knn_model(X_train, X_test, y_train, y_test)
        print("KNN MODEL: ")
        pickle.dump(model, open('../output/knn_model.pkl', 'wb'))
    elif model_type == 'xgboost':
        model = xgboost_model(X_train, y_train, X_test, y_test, X, y, rs)
        print("XGBOOST MODEL: ")
        pickle.dump(model, open(OUTPUT_DIR + 'xgboost_model.pkl', 'wb'))
    else:
        print("Invalid input")
