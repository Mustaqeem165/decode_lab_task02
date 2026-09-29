import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

#  LOAD DATA (FAIL-SAFE FOR CSV OR EXCEL)
def load_data():
    try:
        return pd.read_csv("DATA.csv")
    except Exception:
        try:
            return pd.read_excel("DATA.csv")
        except Exception:
            return pd.read_excel("Dataset for Data Analytics.xlsx")

df = load_data()
print(f"[STEP 1] Data loaded. Initial Shape: {df.shape}")

# Impute missing values
if "CouponCode" in df.columns:
    df["CouponCode"] = df["CouponCode"].fillna("NONE")

# Vectorized Feature Engineering
df["calculated_total"] = df["Quantity"] * df["UnitPrice"]
df["items_per_unit_price"] = df["ItemsInCart"] / (df["UnitPrice"] + 1e-5)
df["quantity_cart_ratio"] = df["Quantity"] / (df["ItemsInCart"] + 1e-5)

# Formulate Imbalanced Fraud Target (High-risk Cancelled Orders)
is_cancelled = df["OrderStatus"] == "Cancelled"
is_high_value = df["TotalPrice"] > df["TotalPrice"].quantile(0.85)
is_cart_spike = df["ItemsInCart"] >= 8
df["is_fraud"] = (is_cancelled & (is_high_value | is_cart_spike)).astype(int)

fraud_count = df["is_fraud"].sum()
print(f" -> Class Distribution: Legitimate={len(df)-fraud_count} ({(1 - df['is_fraud'].mean())*100:.2f}%), Fraud={fraud_count} ({df['is_fraud'].mean()*100:.2f}%)")

# Drop non-predictive identifiers & target-leakage columns
drop_cols = ["OrderID", "CustomerID", "ShippingAddress", "TrackingNumber", "Date", "OrderStatus", "is_fraud"]
X = df.drop(columns=[c for c in drop_cols if c in df.columns])
X = pd.get_dummies(X, drop_first=True, dtype=float)
y = df["is_fraud"]

#  ZERO-LEAKAGE STRATIFIED SPLIT
# Never apply scaling or SMOTE prior to the split!
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"[STEP 2] Stratified Split -> Train: {len(X_train)} samples, Test: {len(X_test)} samples")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#  LINEAR ENGINE: LOGISTIC REGRESSION
print("\n[STEP 3] Tuning Logistic Regression (StandardScaler + SMOTE + LR)...")
lr_pipeline = ImbPipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

grid_lr = GridSearchCV(
    lr_pipeline,
    param_grid={
        "smote__k_neighbors": [3, 5],
        "classifier__C": [0.01, 0.1, 1.0]
    },
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)
grid_lr.fit(X_train, y_train)

# 4. ENSEMBLE TREE ENGINE: RANDOM FOREST
print("[STEP 4] Tuning Random Forest (SMOTE + RF, Scale-Invariant)...")
rf_pipeline = ImbPipeline([
    ("smote", SMOTE(random_state=42)),
    ("classifier", RandomForestClassifier(random_state=42))
])

grid_rf = GridSearchCV(
    rf_pipeline,
    param_grid={
        "smote__k_neighbors": [3, 5],
        "classifier__max_depth": [5, 10, None],
        "classifier__n_estimators": [100, 200]
    },
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)
grid_rf.fit(X_train, y_train)

#  STRICT EVALUATION (NO ACCURACY)
def evaluate_model(model, name: str):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n==========================================")
    print(f" EVALUATION: {name}")
    print(f"==========================================")
    print(f" • Best Parameters  : {model.best_params_}")
    print(f" • Best CV ROC-AUC  : {model.best_score_:.4f}")
    print(f" • Test Recall      : {recall_score(y_test, y_pred):.4f}")
    print(f" • Test Precision   : {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f" • Test F1-Score    : {f1_score(y_test, y_pred):.4f}")
    print(f" • Test ROC-AUC     : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud'])}")

evaluate_model(grid_lr, "Tuned Logistic Regression")
evaluate_model(grid_rf, "Tuned Random Forest")