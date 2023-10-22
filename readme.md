# Real Estate Price Prediction

## Project Overview

### Objective
In this project, we aim to predict house prices for 200 apartments in Pune city. We will use various regression models, such as Linear Regression, Random Forest, XGBoost, and multi-layer perceptron (MLP) models using scikit-learn and TensorFlow. The goal is to help predict house prices based on different property features.

---

### Data
We have a dataset with around 200 rows and 17 variables that influence the target variable, which is the house price.

---

### Tech Stack
- Language: `Python`
- Libraries: `scikit-learn`, `pandas`, `NumPy`, `matplotlib`, `seaborn`, `xgboost`

---

## Project Phases

### 1. Data Cleaning
- Import required libraries and load the dataset.
- Perform preliminary data exploration.
- Identify and remove outliers.
- Remove redundant feature columns.
- Handle missing values.
- Regularize categorical columns.
- Save the cleaned data.

### 2. Data Analysis
- Import the cleaned dataset.
- Convert binary columns to dummy variables.
- Perform feature engineering.
- Conduct univariate and bivariate analysis.
- Check for correlations.
- Select relevant features.
- Scale the data.
- Save the final updated dataset.

### 3. Model Building
- Prepare the data.
- Split the dataset into training and testing sets.
- Build various regression models, including Linear Regression, Ridge Regression, Lasso Regressor, Elastic Net, Random Forest Regressor, XGBoost Regressor, K-Nearest Neighbours Regressor, and Support Vector Regressor.

### 4. Model Validation
- Assess model performance using metrics like Mean Squared Error (MSE) and R2 score.
- Create residual plots for both training and testing data.

### 5. Hyperparameter Tuning
- Perform grid search and cross-validation for the chosen regressor.

### 6. Making Predictions
- Fit the model and make predictions on the test data.

### 7. Feature Importance
- Check for feature importance to identify the most influential factors in predicting house prices.

### 8. Model Comparison
- Compare the performance of different models to choose the best one.

### 9. MLP (Multi-Layer Perceptron) Models
- Build MLP Regression models using both scikit-learn and TensorFlow.

---

## Getting Started

To run this project, follow these steps:

1. Install the required libraries listed in `requirements.txt`.
2. Execute the code in the `src` folder for each project phase.

---

## Concepts Explored

1. Basic Exploratory Data Analysis (EDA)
2. Data cleaning and missing data handling
3. Checking for outliers
4. Matplotlib and seaborn for data interpretation and advanced visualizations.
5. Feature Engineering on data for better performance.
6. Regression techniques like Linear Regression, Random Forest Regressor, XGBoost Regressor, etc.
7. Grid search and Cross-validation for the given regressor.
8. Predictions using the trained model.
9. Metrics such as MSE, R2
10. Residual plots for train and test data
11. Feature Importance.
12. Model comparison
13. Multi-Layer Perceptron model using the Scikit-learn library.
14. Multi-Layer Perceptron model using TensorFlow.

---
