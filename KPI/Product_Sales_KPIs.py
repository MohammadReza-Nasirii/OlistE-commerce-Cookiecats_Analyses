import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load merged dataset
df = pd.read_csv('../Data/merged_df.cv')

# Calculate total revenue per product
top_products = (
    df.groupby('product_id')
    .agg({'price': 'sum'})
    .rename(columns={'price': 'Total_Revenue'})
    .sort_values(by='Total_Revenue', ascending=False)
    .head(10)
    .reset_index()
)

print("🏆 Top Performing Products:")
print(top_products)

# --- profit margin per product ---
# Calculate total revenue and profit per product
profit_margin_df = df.groupby('product_id').agg({
    'price': 'sum',
    'freight_value': 'sum'
    }).reset_index()

# Calculate total revenue and profit per product
profit_margin_df['profit_margin_%'] = ((profit_margin_df['price'] - profit_margin_df['freight_value']) / profit_margin_df['price']) * 100

# Sort by profit margin
profit_margin_df = profit_margin_df.sort_values(by='profit_margin_%', ascending=False)


# Display top 10 products by profit margin
print("\n🏆 Top 10 Products by Profit Margin (%):")
print(profit_margin_df.head(10))

# --- Sales Velocity (How fast sales are generated) ---

# Convert order purchase timestamp to datetime
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])

# Calculate total duration of dataset in days
sales_duration_days = (df['order_purchase_timestamp'].max() - df['order_purchase_timestamp'].min()).days

# Calculate total sales
total_sales = df['price'].sum()

# Calculate sales velocity
sales_velocity = total_sales / sales_duration_days

print(f"📈 Total Sales Duration (Days): {sales_duration_days}")
print(f"⚡ Sales Velocity (Revenue per Day): {sales_velocity:,.2f}")

# --- Revenue by Category & Region ---

# Group by category and state to get total revenue
revenue_by_region = (
    df.groupby(['product_category_name', 'customer_state'])['price']
    .sum()
    .reset_index()
    .sort_values(by='price', ascending=False)
)

print("\n🏆 Top 10 Category-Region combinations by Revenue:")
print(revenue_by_region.head(10))
