import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib
from matplotlib import style
matplotlib.use('TkAgg')

# Load datasets
orders = pd.read_csv('../Data/raw/olist_orders_dataset.csv')
order_items = pd.read_csv('../Data/raw/olist_order_items_dataset.csv')
products = pd.read_csv('../Data/raw/olist_products_dataset.csv')
customers = pd.read_csv('../Data/raw/olist_customers_dataset.csv')
order_payments = pd.read_csv('../Data/raw/olist_order_payments_dataset.csv')

# Convert date columns
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])

# print("✅ Data Loaded Successfully!")
# print(orders.head(2))

# Merge order_items with orders
merged_df = pd.merge(order_items, orders, on='order_id', how='inner')

# Merge with payments
merged_df = pd.merge(merged_df, order_payments, on='order_id', how='left')

# Merge with products
merged_df = pd.merge(merged_df, products, on='product_id', how='left')

# Merge with customers
merged_df = pd.merge(merged_df, customers, on='customer_id', how='left')

# print("✅ Final Merged Dataset:", merged_df.shape)

merged_df['order_year_month'] = merged_df['order_purchase_timestamp'].dt.to_period('M')
merged_df['order_month'] = merged_df['order_purchase_timestamp'].dt.month
merged_df['order_year'] = merged_df['order_purchase_timestamp'].dt.year

monthly_revenue = merged_df.groupby('order_year_month')['payment_value'].sum()

aov = merged_df.groupby('order_year_month')['payment_value'].mean()

plt.figure(figsize=(10,5))
monthly_revenue.plot(kind='line', marker='o', color='teal', label='Monthly Revenue')
plt.title('Monthly Revenue Trend')
plt.xlabel('Year-Month')
plt.ylabel('Total Revenue (BRL)')
plt.legend()
plt.tight_layout()
plt.show()

top_categories = merged_df.groupby('product_category_name')['payment_value'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,5))
sns.barplot(x=top_categories.values, y=top_categories.index, palette='viridis')
plt.title('Top 10 Product Categories by Revenue')
plt.xlabel('Total Revenue (BRL)')
plt.tight_layout()
plt.show()

print("📊 Monthly Revenue Summary:")
print(monthly_revenue.describe())

print("\n💰 Average Order Value per Month:")
print(aov.tail(12))