import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use("TkAgg")

# ---------------------------
# 1) Load merged data
# ---------------------------
df = pd.read_csv("../../Data/processed/merged_df.csv")

# ---------------------------
# 2) Parse datetime columns
# ---------------------------
df["order_purchase_timestamp"] = pd.to_datetime(
    df["order_purchase_timestamp"], errors="coerce"
)
df["order_delivered_customer_date"] = pd.to_datetime(
    df["order_delivered_customer_date"], errors="coerce"
)

# در صورت وجود ردیف‌های بدون تاریخ خرید، حذفشان می‌کنیم
df = df.dropna(subset=["order_purchase_timestamp"])

# ---------------------------
# 3) Aggregate per customer
# ---------------------------
customer_df = df.groupby("customer_unique_id").agg({
    "price": "sum",
    "order_id": "nunique",
    "order_purchase_timestamp": ["min", "max"]
}).reset_index()

customer_df.columns = [
    "customer_id",
    "total_spent",
    "num_orders",
    "first_purchase",
    "last_purchase",
]

# ---------------------------
# 4) Create churn label (business rule)
# ---------------------------
# current_date = آخرین تاریخ خرید در دیتاست
current_date = df["order_purchase_timestamp"].max()

customer_df["customer_age_days"] = (
    customer_df["last_purchase"] - customer_df["first_purchase"]
).dt.days

customer_df["recency_days"] = (
    current_date - customer_df["last_purchase"]
).dt.days

# تعریف churn: اگر بیش از 180 روز از آخرین خرید گذشته باشد
CHURN_THRESHOLD = 180
customer_df["churn"] = np.where(
    customer_df["recency_days"] > CHURN_THRESHOLD, 1, 0
)

# ---------------------------
# 5) Quick sanity check
# ---------------------------
churn_distribution = (
    customer_df["churn"]
    .value_counts(normalize=True)
    .rename("proportion")
)
print("Churn distribution:")
print(churn_distribution, "\n")

# ---------------------------
# 6) Prepare data for modeling
# ---------------------------
feature_cols = [
    "total_spent",
    "num_orders",
    "avg_order_value",
    "recency_days",
    "customer_age_days",
]

# Average Order Value
customer_df["avg_order_value"] = (
    customer_df["total_spent"] / customer_df["num_orders"]
)

X = customer_df[feature_cols]
y = customer_df["churn"]

# Train / Test split (random, but stratified by churn label)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# ---------------------------
# 7) Train Random Forest model
# ---------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ---------------------------
# 8) Evaluation
# ---------------------------
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {acc:.4f}\n")

print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Predicted 0", "Predicted 1"],
    yticklabels=["Actual 0", "Actual 1"],
)
plt.title("Customer Churn Prediction – Confusion Matrix")
plt.tight_layout()
plt.show()

# ---------------------------
# 9) Feature importance
# ---------------------------
feat_importance = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": model.feature_importances_,
}).sort_values(by="Importance", ascending=False)

print("\n🔍 Feature Importance:")
print(feat_importance)

# ---------------------------
# 10) Optional: churn risk segments
# ---------------------------
# احتمال churn هر مشتری (probability for class 1)
customer_df["churn_prob"] = model.predict_proba(X)[:, 1]

# تقسیم به دو سگمنت ساده: High risk / Low risk
customer_df["churn_risk_segment"] = np.where(
    customer_df["churn_prob"] > 0.5, "High", "Low"
)

segment_summary = customer_df.groupby("churn_risk_segment").agg(
    num_customers=("customer_id", "count"),
    avg_total_spent=("total_spent", "mean"),
    avg_num_orders=("num_orders", "mean"),
    avg_churn_prob=("churn_prob", "mean"),
)

print("\n📌 Churn Risk Segments Summary:")
print(segment_summary)
