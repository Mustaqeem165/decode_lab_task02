# Imbalanced Fraud Detection Pipeline

This project is a comprehensive machine learning pipeline designed to detect fraudulent e-commerce transactions using advanced classification and oversampling techniques. The script (`project2_decode.py`) processes raw transaction data, engineers features, handles severe class imbalances using SMOTE, and evaluates models using strict performance metrics.

## Core Features

* **Robust Data Processing:** Automatically ingests datasets (defaulting to `DATA.csv` with Excel fallbacks), imputes missing categorical values, and generates vectorized features such as total cost and cart-to-quantity ratios.


* **Target Formulation:** Programmatically defines an imbalanced `is_fraud` target variable, identifying high-risk transactions as those that are cancelled and either have a top-15% total price or contain 8 or more items in the cart.


* **Zero-Leakage Stratified Splitting:** Prevents data leakage by dropping highly correlated or predictive identifiers (like `OrderStatus`) and separating the data into an 80/20 train-test split before applying any scaling or oversampling, maintaining the exact fraud-to-legitimate class distribution via stratification.


* **Imbalanced Learning Pipelines:** Utilizes the `imblearn` library to construct two robust cross-validation pipelines:
* **Linear Engine:** Standard Scaler + SMOTE + Logistic Regression.


* **Ensemble Tree Engine:** SMOTE + Random Forest Classifier (Scale-Invariant).




* **Automated Hyperparameter Tuning:** Employs `GridSearchCV` with 5-fold Stratified K-Fold cross-validation to find the optimal parameters (like SMOTE nearest neighbors, LR regularization, and RF depth) by optimizing for the ROC-AUC score.


* **Strict Metric Evaluation:** Explicitly avoids misleading accuracy scores for imbalanced datasets, evaluating the final models on the test set using Recall, Precision, F1-Score, ROC-AUC, and comprehensive confusion matrices.



## Requirements

The pipeline requires the following Python libraries:

* `numpy`
* `pandas`
* `scikit-learn`
* `imbalanced-learn`

## Execution

Ensure your dataset is present in the working directory as `DATA.csv` and run the script. The console will output the initial data shape, the engineered class distribution, the progress of the Grid Search optimizations, and a final detailed evaluation report for both the Logistic Regression and Random Forest models.
