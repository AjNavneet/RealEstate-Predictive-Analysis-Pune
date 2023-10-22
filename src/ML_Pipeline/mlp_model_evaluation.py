# Import the required libraries
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
def gridSearchReportMLP(estimator, X, y, pg, cv=LeaveOneOut(), rs=118):
    """
    Performs the grid search and cross-validation for the given regressor.
    Parameters:
        estimator: The regressor
        X: Pandas dataframe, feature data
        y: Pandas series, target
        pg: Dict, parameters' grid
        cv: Int or cross-validation generator or an iterable, cross-validation folds
        rs: Int, training-test split random state
    """

    t0 = time()
    
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=.3, random_state=rs)

    est_cv = GridSearchCV(
        estimator, 
        param_grid=pg, 
        scoring="neg_mean_squared_error", 
        n_jobs=-1, 
        cv=cv
    )
    
    est_cv.fit(X_train, y_train)
    
    print("Best parameters:", est_cv.best_params_)
    print("Best CV score:", abs(est_cv.best_score_))
    y_train_pred, y_test_pred = est_cv.predict(X_train), est_cv.predict(X_test)
    print("MSE, R2 train:", mean_squared_error(y_train, y_train_pred), ", ", r2_score(y_train, y_train_pred))
    print("MSE, R2 test:", mean_squared_error(y_test, y_test_pred), ", ", r2_score(y_test, y_test_pred))
    
    t = round(time()-t0, 2)
    print("Elapsed time:", t, "s, ", round(t/60, 2), "min")
    
    return est_cv

# Plot residuals for MLP model
def plotResidueMLP(estimator, X, y, rs=118):
    """
    Plots the fit residuals (price - predicted_price) vs. "surface" variable.
    Parameters:
        estimator: GridSearchCV, the regressor
        X: Pandas dataframe, feature data
        y: Pandas series, target
        rs: Int, random state
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=rs) 
    
    residue_train = y_train.values.reshape(-1,1)-estimator.predict(X_train.values).reshape(-1,1)
    residue_test = y_test.values.reshape(-1,1)-estimator.predict(X_test.values).reshape(-1,1)                                                     
               
    fig, axe = plt.subplots(1, 2, figsize=(18,10)) 
    axe[0].scatter(X_train["surface"], residue_train, label="train")
    axe[0].scatter(X_test["surface"], residue_test, label="test")
    axe[0].plot([-2.3, 4.5], [0,0], "black")
    axe[0].set_xlabel("Scaled surface")
    axe[0].set_ylabel("Fit residuals")
    axe[0].legend()
    
    axe[1].hist(residue_test, bins=25)
    axe[1].set_xlabel("Fit residual for the test set")
    axe[1].set_ylabel("Count")

    plt.savefig(OUTPUT_DIR + "plot_residual_mlp")

    print("Mean residuals:", round(np.mean(residue_test), 2), "\nStandard deviation:", round(np.std(residue_test), 2))

# Plot real vs. predicted prices
def plot_real_pred(est, X, y, rs):
    """
    Plots the real price vs. predicted price
    Parameters:
        est: The regressor
        X: Pandas dataframe, feature data
        y: Pandas series, target
        rs: Int, random state
    """
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=.3, random_state=rs)
    
    fig = plt.figure(figsize=(7,7))
    plt.scatter(y_test, est.predict(X_test))
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], c="k")
    plt.xlabel("Real price")
    plt.ylabel("Predicted price")
    plt.savefig(OUTPUT_DIR + "plot_real_pred_mlp")

# Plot the history of the TensorFlow network
def plot_model_history(history):
    """
    Plot the training and validation history for a TensorFlow network.
    Parameters:
        history: The history object of the TensorFlow model.
    """

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['mae']
    val_acc = history.history['val_mae']
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
    ax[0].plot(np.arange(len(loss)), loss, label='Training')
    ax[0].plot(np.arange(len(val_loss)), val_loss, label='Validation')
    ax[0].set_title('Mean Square Error')
    ax[0].legend()
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')

    ax[1].plot(np.arange(len(acc)), acc, label='Training')
    ax[1].plot(np.arange(len(acc)), val_acc, label='Validation')
    ax[1].set_title('Mean Absolute Error Curves')
    ax[1].legend()
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Mean Absolute Error')
    plt.savefig(OUTPUT_DIR + "mlp_history")
