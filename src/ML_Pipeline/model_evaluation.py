# import the required libraries
import numpy as np
import pandas as pd
from time import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error
import configparser

config = configparser.RawConfigParser()
config.read('..\\input\\config.ini')
OUTPUT_DIR = config.get('DATA', 'output_dir')

# Function for performing grid search
def gridSearchReport(estimator, X, y, pg, cv=LeaveOneOut(), rs=118):
    """
    Performs the grid search and cross-validation for the given regressor.
    Params:
        estimator: the regressor
        X: Pandas dataframe, feature data
        y: Pandas series, target
        pg: dict, parameters' grid
        cv: int, cross-validation generator or an iterable, cross-validation folds
        rs: int, training-test split random state
    """
    t0 = time()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=rs) # Train-test split for normal regression models

    est_cv = GridSearchCV(
        estimator,
        param_grid=pg,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        cv=cv
    )

    est_cv.fit(X_train, y_train)

    print("Best parameters:", est_cv.best_params_) # Best parameters
    print("Best CV score:", abs(est_cv.best_score_)) # Cross-validation score
    y_train_pred, y_test_pred = est_cv.predict(X_train), est_cv.predict(X_test) # Predictions on train and test data
    print("MSE, R2 train:", mean_squared_error(y_train, y_train_pred), # Evaluation metrics
          ", ", r2_score(y_train, y_train_pred))
    print("MSE, R2 test:", mean_squared_error(y_test, y_test_pred),
          ", ", r2_score(y_test, y_test_pred))

    t = round(time()-t0, 2)
    print("Elapsed time:", t, "s ,", round(t/60, 2), "min")

    return est_cv

# Function to plot the residuals
def plotResidue(estimator, X, y, rs=118):
    """
    Plots the fit residuals (price - predicted_price) vs. "surface" variable.
    Params:
        estimator: GridSearchCV, the regressor
        X: Pandas dataframe, feature data
        y: Pandas series, target
        rs: int, random state
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=rs) # Train-test split

    residue_train = y_train - estimator.predict(X_train) # Make predictions
    residue_test = y_test - estimator.predict(X_test)

    fig, axe = plt.subplots(1, 2, figsize=(18, 10))
    axe[0].scatter(X_train["surface"], residue_train, label="train")
    axe[0].scatter(X_test["surface"], residue_test, label="test")
    axe[0].plot([-2.3, 4.5], [0, 0], "black")
    axe[0].set_xlabel("Scaled surface")
    axe[0].set_ylabel("Fit residuals")
    axe[0].legend()

    axe[1].hist(residue_test, bins=25)
    axe[1].set_xlabel("Fit residual for the test set")
    axe[1].set_ylabel("Count")

    #plt.show()
    plt.savefig(OUTPUT_DIR + "plot_residual.png")
    print("mean residuals:", round(np.mean(residue_test), 2),
          "\nstd:", round(np.std(residue_test), 2))
