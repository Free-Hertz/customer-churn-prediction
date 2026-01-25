# Customer Churn Prediction

## Overview
Customer churn refers to customers discontinuing a subscription-based service. Retaining existing customers is often more cost-effective than acquiring new ones.
The goal of this project is to build a machine learning model that predicts whether a customer is likely to churn, enabling businesses to take proactive retention actions.

## Dataset
The dataset contains customer demographics, account information, and service usage details for a telecom company.

## Target Variable:
Churn Label (1 = Churn, 0 = No Churn)

## Features include:
Demographics (Age, Gender, Dependents, etc.)
Account details (Contract Type, Payment Method, Tenure, etc.)
Service usage (Internet Service, Streaming Services, Charges, etc.)

## Approach
1. Data Loading & Inspection
2. Exploratory Data Analysis (EDA)
3. Data Cleaning
    - Dropped leakage and irrelevant columns
    - Handled missing values
4. Feature Encoding
    - Label encoding for target
    - One-hot encoding for categorical features
5. Train-Test Split (Stratified)
6. Feature Scaling
7. Model Training
    - Logistic Regression (baseline)
    - Random Forest
    - Logistic Regression with class weighting
8. Model Evaluation
    - Accuracy
    - Precision, Recall, F1-score
    - Confusion Matrix
9. Feature Importance Analysis
10. Final Model Selection

## Models Used
-Logistic Regression
-Random Forest Classifier
-Logistic Regression with class_weight='balanced'

Balanced Logistic Regression was selected as the final model due to it's improved recall for churned customers.

## Results
Final Model: Balanced Logistic Regression
Accuracy ~ 76%
Recall (Churn Class) ~ 64%

The model prioritizes identifying potential churners, which aligns with business goals.

## Key Insights
-Customers with short tenure are more likely to churn
-Month-to-month contracts show higher churn rates
-Longer contracts and bundled services reduce churn probability

## Technologies Used
-Python
-Pandas
-NumPy
-Scikit-learn
-Matplotlib
-Seaborn
-Jupyter Notebook

## How to Run
Clone the repository
Install required libraries
    -pip install -r requirements.txt

## Open the Jupyter Notebook
Run all cells from top to bottom

## Future Improvements
Hyperparameter tuning
Try advanced models (XGBoost, LightGBM)
More feature engineering
Outlier treatment
Model deployment as a web application

## Author
Harbaz Singh
