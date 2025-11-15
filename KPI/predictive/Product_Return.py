import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load merged dataset
df = pd.read_csv('../../Data/processed/merged_df.csv')

# -----------------------------
# 1. Convert dates to datetime
# -----------------------------
date_cols = [
    'order_purchase_timestamp',
    'order_approved_at',
    'order_delivered_carrier_date',
    'order_delivered_customer_date',
    'order_estimated_delivery_date'
]

for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')


# -----------------------------
# 2. Build Return Label
# -----------------------------
# Rule 1: canceled orders = return
df['is_return'] = df['order_status'].apply(lambda x: 1 if x == "canceled" else 0)

# Rule 2: review score 1 or 2 = return
if 'review_score' in df.columns:
    df['is_return'] = df.apply(
        lambda row: 1 if row.get('review_score', 5) <= 2 else row['is_return'],
        axis=1
    )
else:
    print("⚠ Warning: review_score not found in merged dataset")


# -----------------------------
# 3. Keep only useful features
# -----------------------------
model_df = df[[
    'price', 'freight_value',
    'product_weight_g', 'product_length_cm',
    'product_height_cm', 'product_width_cm',
    'customer_state', 'product_category_name',
    'is_return'
]].dropna()

print("📊 Data prepared for Return Model:")
print(model_df.head())

# ------------------------------------
# Encode categorical features
# ------------------------------------
cat_cols = ['customer_state', 'product_category_name']

encoder = LabelEncoder()
for col in cat_cols:
    model_df[col] = encoder.fit_transform(model_df[col])

# ------------------------------------
# Train/Test Split
# ------------------------------------
X = model_df.drop('is_return', axis=1)
y = model_df['is_return']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ------------------------------------
# Model Training (Random Forest)
# ------------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',   # VERY IMPORTANT for imbalanced data
    random_state=42
)

model.fit(X_train, y_train)

# ------------------------------------
# Predictions
# ------------------------------------
y_pred = model.predict(X_test)

# ------------------------------------
# Evaluation Metrics
# ------------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n✅ Random Forest Return Prediction Results:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# ------------------------------------
# Feature Importance
# ------------------------------------
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values(by='importance', ascending=False)

print("\n📊 Feature Importances:")
print(importances)

# -----------------------------
# 4. Encode categorical features
# -----------------------------
cat_cols = ['customer_state', 'product_category_name']

encoder = LabelEncoder()

for col in cat_cols:
    model_df[col] = encoder.fit_transform(model_df[col].astype(str))

# -----------------------------
# 5. Split data: X / y
# -----------------------------
X = model_df.drop('is_return', axis=1)
y = model_df['is_return']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 6. Train Random Forest model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight='balanced'   # VERY IMPORTANT (because return is rare)
)

model.fit(X_train, y_train)

# -----------------------------
# 7. Predictions & Evaluation
# -----------------------------
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n✅ Random Forest Return Prediction Model Results:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")

# -----------------------------
# 8. Feature Importance
# -----------------------------
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\n📊 Feature Importances:")
print(feature_importance)