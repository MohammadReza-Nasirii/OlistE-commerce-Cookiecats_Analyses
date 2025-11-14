# KPI/Predictive/high_value_propensity.py
# Predict which customers are likely to be High-Value using XGBoost (classification)

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix
)
from xgboost import XGBClassifier

# -----------------------------
# 1) Load merged data
# -----------------------------
# Adjust the path to your processed merged dataset
DATA_PATH = Path("../Data/processed/merged_df.csv")
df = pd.read_csv(DATA_PATH)

# Basic sanity checks
required_cols = [
    "customer_unique_id",
    "order_id",
    "order_purchase_timestamp",
    "price"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in merged_df.csv: {missing}")

# -----------------------------
# 2) Prepare timestamps
# -----------------------------
df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"], errors="coerce")
df = df.dropna(subset=["order_purchase_timestamp"])

# -----------------------------
# 3) Build customer-level table
# -----------------------------
customer = (
    df.groupby("customer_unique_id")
      .agg(
          total_spent=("price", "sum"),
          num_orders=("order_id", "nunique"),
          first_purchase=("order_purchase_timestamp", "min"),
          last_purchase=("order_purchase_timestamp", "max"),
      )
      .reset_index()
)

# Derived features
current_date = df["order_purchase_timestamp"].max()
customer["customer_age_days"] = (customer["last_purchase"] - customer["first_purchase"]).dt.days
customer["recency_days"] = (current_date - customer["last_purchase"]).dt.days
customer["avg_order_value"] = customer["total_spent"] / customer["num_orders"].replace(0, np.nan)

# Handle potential NaNs (e.g., if num_orders == 0 which is rare after nunique on orders)
customer = customer.replace([np.inf, -np.inf], np.nan)
customer = customer.fillna({
    "avg_order_value": 0.0,
    "customer_age_days": 0,
    "recency_days": 0
})

# -----------------------------
# 4) Create High-Value label (top 20% by total_spent)
# -----------------------------
threshold = customer["total_spent"].quantile(0.80)
customer["high_value"] = (customer["total_spent"] >= threshold).astype(int)

# -----------------------------
# 5) Feature matrix and target
# -----------------------------
feature_cols = ["total_spent", "num_orders", "avg_order_value", "recency_days", "customer_age_days"]
X = customer[feature_cols].copy()
y = customer["high_value"].copy()

# Optional: scale/normalize not required for tree-based models
# Ensure numeric types
X = X.astype(float)

# -----------------------------
# 6) Train / Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# -----------------------------
# 7) Train XGBoost classifier
# -----------------------------
model = XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="binary:logistic",
    eval_metric="auc",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# -----------------------------
# 8) Evaluate
# -----------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_proba)

print("✅ High-Value Customer Propensity — XGBoost")
print(f"Accuracy:  {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall:    {rec:.3f}")
print(f"ROC-AUC:   {auc:.3f}\n")

print("📊 Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("🔢 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# 9) Feature Importance table
# -----------------------------
importances = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\n🏗 Feature Importances:")
print(importances.to_string(index=False))

# -----------------------------
# 10) (Optional) Save outputs
# -----------------------------
OUTPUT_DIR = Path("../Outputs/predictive")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
importances.to_csv(OUTPUT_DIR / "high_value_feature_importance.csv", index=False)
customer[["customer_unique_id", "total_spent", "num_orders", "avg_order_value",
          "recency_days", "customer_age_days", "high_value"]].to_csv(
    OUTPUT_DIR / "high_value_training_table.csv", index=False
)

print("\n💾 Saved: feature importances and training table in ../Outputs/predictive/")
