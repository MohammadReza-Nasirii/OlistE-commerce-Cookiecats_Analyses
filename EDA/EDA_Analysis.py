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
merged_df['year'] = merged_df['order_purchase_timestamp'].dt.year
merged_df['month'] = merged_df['order_purchase_timestamp'].dt.month
merged_df['day'] = merged_df['order_purchase_timestamp'].dt.day
merged_df['day_of_week'] = merged_df['order_purchase_timestamp'].dt.day_name()
merged_df['total_sales'] = merged_df['price'] + merged_df['freight_value']

merged_df.to_csv('../Data/merged_df.cv')

monthly_revenue = merged_df.groupby('order_year_month')['payment_value'].sum()

aov = merged_df.groupby('order_year_month')['payment_value'].mean()

plt.figure(figsize=(10, 5))
monthly_revenue.plot(kind='line', marker='o', color='teal', label='Monthly Revenue')
plt.title('Monthly Revenue Trend')
plt.xlabel('Year-Month')
plt.ylabel('Total Revenue (BRL)')
plt.legend()
plt.tight_layout()
plt.show()

top_categories = merged_df.groupby('product_category_name')['payment_value'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 5))
sns.barplot(x=top_categories.values, y=top_categories.index, palette='viridis')
plt.title('Top 10 Product Categories by Revenue')
plt.xlabel('Total Revenue (BRL)')
plt.tight_layout()
plt.show()

print("📊 Monthly Revenue Summary:")
print(monthly_revenue.describe())

print("\n💰 Average Order Value per Month:")
print(aov.tail(12))

top_customers = merged_df.groupby('customer_id')['price'].sum().sort_values(ascending=False).head(10)
print("\n💰 Top 10 Customers by Total Spending:")
print(top_customers)

top_products = merged_df.groupby('product_id')['price'].sum().sort_values(ascending=False).head(10)
print("\n🏆 Top 10 Products by Total Revenue:")
print(top_products)

top_categories = merged_df.groupby('product_category_name')['price'].sum().sort_values(ascending=False).head(10)
print("\n📦 Top 10 Product Categories by Revenue:")
print(top_categories)

plt.figure(figsize=(10, 5))
top_customers.plot(kind='bar', color='skyblue')
plt.title("Top 10 Customers by Total Spending")
plt.xlabel("Customer ID")
plt.ylabel("Total Spending")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
top_customers.plot(kind='bar', color='skyblue')
plt.title("Top 10 Customers by Total Spending")
plt.xlabel("Customer ID")
plt.ylabel("Total Spending")
plt.tight_layout()
plt.show()

customer_sales = merged_df.groupby('customer_id')['price'].sum().sort_values(ascending=False).reset_index()
customer_sales['cumulative_sales'] = customer_sales['price'].cumsum()
customer_sales['cumulative_perc'] = 100 * customer_sales['cumulative_sales'] / customer_sales['price'].sum()

pareto_cutoff_customer = customer_sales[customer_sales['cumulative_perc'] <= 80]

print("\n📊 Pareto Analysis (Customers):")
print(f"Top {len(pareto_cutoff_customer)} customers (~{len(pareto_cutoff_customer) / len(customer_sales) * 100:.1f}"
      f"%) generate 80% of total revenue.")

product_sales = merged_df.groupby('product_id')['price'].sum().sort_values(ascending=False).reset_index()
product_sales['cumulative_sales'] = product_sales['price'].cumsum()
product_sales['cumulative_perc'] = 100 * product_sales['cumulative_sales'] / product_sales['price'].sum()

pareto_cutoff_products = product_sales[product_sales['cumulative_perc'] <= 80]

print("\n📦 Pareto Analysis (Products):")
print(
    f"Top {len(pareto_cutoff_products)} products (~{len(pareto_cutoff_products) / len(product_sales) * 100:.1f}%) generate 80% of total revenue.")

plt.figure(figsize=(10, 5))
plt.plot(customer_sales['cumulative_perc'], color='blue', label='Cumulative % of Revenue')
plt.axhline(80, color='red', linestyle='--', label='80% Threshold')
plt.title('Pareto Analysis - Customer Revenue Distribution')
plt.xlabel('Customers (sorted by revenue)')
plt.ylabel('Cumulative % of Total Revenue')
plt.legend()
plt.tight_layout()
plt.show()

monthly_sales = merged_df.groupby(['year', 'month'])['price'].sum().reset_index()

monthly_sales['year_month'] = pd.to_datetime(monthly_sales[['year', 'month']].assign(day=1))

plt.figure(figsize=(12, 6))
sns.lineplot(x='year_month', y='price', data=monthly_sales, marker='o')
plt.title('Monthly Sales Trend')
plt.xlabel('Date')
plt.ylabel('Total Sales (Revenue)')
plt.grid(True)
plt.tight_layout()
plt.show()

weekly_sales = merged_df.groupby('day_of_week')['price'].sum().reindex(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
).reset_index()

plt.figure(figsize=(8, 5))
sns.barplot(x='day_of_week', y='price', data=weekly_sales, palette='viridis')
plt.title('Weekly Sales Distribution')
plt.xlabel('Day of Week')
plt.ylabel('Total Sales')
plt.tight_layout()
plt.show()


def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers


# Detect outliers for Sales and Profit
outliers_sales = detect_outliers_iqr(merged_df, 'price')

outliers_profit = detect_outliers_iqr(merged_df, 'freight_value')

print(f"Number of outliers in Sales: {len(outliers_sales)}")
print(f"Number of outliers in Profit: {len(outliers_profit)}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(data=merged_df, x='price', color='skyblue')
plt.title('Price Distribution with Outliers')

plt.subplot(1, 2, 2)
sns.boxplot(data=merged_df, x='freight_value', color='lightgreen')
plt.title('Freight Value Distribution with Outliers')

plt.tight_layout()
plt.show()

outlier_ratio_price = len(outliers_sales) / len(merged_df) * 100
outlier_ratio_freight = len(outliers_profit) / len(merged_df) * 100


# print(f"Outliers in Price: {outlier_ratio_price:.2f}%")
# print(f"Outliers in Freight Value: {outlier_ratio_freight:.2f}%")

def winsorize_column(df, column, lower_percentile=0.05, upper_percentile=0.95):
    """Cap values below 5th and above 95th percentile."""
    lower_limit = df[column].quantile(lower_percentile)
    upper_limit = df[column].quantile(upper_percentile)
    df[column] = np.clip(df[column], lower_limit, upper_limit)
    return df


# Apply to key numerical columns
merged_df = winsorize_column(merged_df, 'price')
merged_df = winsorize_column(merged_df, 'freight_value')

print("✅ Winsorizing complete.")
print(f"Price range after capping: {merged_df['price'].min()} - {merged_df['price'].max()}")
print(f"Freight range after capping: {merged_df['freight_value'].min()} - {merged_df['freight_value'].max()}")

numeric_cols = merged_df.select_dtypes(include=['float64', 'int64']).columns

corr_matrix = merged_df[numeric_cols].corr()

print("✅ Correlation Matrix:")
print(corr_matrix[['price', 'freight_value', 'payment_value']].head())

plt.figure(figsize=(10, 6))
sns.heatmap(
    corr_matrix,
    cmap='coolwarm',
    annot=False,
    center=0
)
plt.title('Correlation Heatmap - Numerical Features')
plt.tight_layout()
plt.show()

selected_features = ['price', 'freight_value', 'payment_value', 'product_weight_g']
plt.figure(figsize=(6, 4))
sns.heatmap(
    merged_df[selected_features].corr(),
    cmap='YlGnBu',
    annot=True,
    fmt=".2f"
)
plt.title('Focused Correlation Between Key Financial Variables')
plt.tight_layout()
plt.show()