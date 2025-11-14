# --- Import Libraries ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')

# --- Load data (use your merged dataset with customer info) ---
df = pd.read_csv('../Data/processed/merged.csv')

# --- Feature Engineering ---
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])

# Calculate recency per customer
customer_df = df.groupby('customer_unique_id').agg({
    'price': 'sum',
    'order_id': 'nunique',
    'order_purchase_timestamp': ['min', 'max']
}).reset_index()

customer_df.columns = ['customer_id', 'total_spent', 'num_orders', 'first_purchase', 'last_purchase']
current_date = df['order_purchase_timestamp'].max()

# Derived features
customer_df['customer_age_days'] = (customer_df['last_purchase'] - customer_df['first_purchase']).dt.days
customer_df['recency_days'] = (current_date - customer_df['last_purchase']).dt.days
customer_df['avg_order_value'] = customer_df['total_spent'] / customer_df['num_orders']

# Define churn: inactive for >180 days
customer_df['churn'] = np.where(customer_df['recency_days'] > 180, 1, 0)

# --- Prepare data for model ---
X = customer_df[['total_spent', 'num_orders', 'avg_order_value', 'recency_days', 'customer_age_days']]
y = customer_df['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Train Random Forest Classifier ---
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# --- Evaluate ---
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {acc:.2f}")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# --- Confusion Matrix ---
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Customer Churn Prediction Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# --- Feature Importance ---
feat_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\n🔍 Feature Importance:")
print(feat_importance)
