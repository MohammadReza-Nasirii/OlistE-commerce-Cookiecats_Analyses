import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Load merged dataset ---
df = pd.read_csv('../../Data/processed/merged_df.csv')

# --- Select features for sales prediction ---
features = ['freight_value', 'product_weight_g', 'product_length_cm',
            'product_height_cm', 'product_width_cm']

target = 'price'

# Drop missing values
df = df.dropna(subset=features + [target])

X = df[features]
y = df[target]

# --- Split into train/test ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train model ---
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Predict ---
preds = model.predict(X_test)

# --- Evaluate ---
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print(f"✅ Random Forest Model Evaluation:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# --- Feature Importance ---
importances = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\n📊 Feature Importances:")
print(importances)
