import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
matplotlib.use('TkAgg')

df = pd.read_csv('../Data/merged_df.cv')

date_cols = [
    'order_purchase_timestamp',
    'order_delivered_customer_date',
    'order_approved_at',
    'order_estimated_delivery_date'
]

for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')


print(df[['order_id', 'customer_id', 'price', 'order_purchase_timestamp', 'order_delivered_customer_date']].head())

total_revenue = df['price'].sum()

total_orders = df['order_id'].nunique()
aov = total_revenue / total_orders

df['order_year_month'] = df['order_purchase_timestamp'].dt.to_period('M')

customers_by_month = df.groupby('order_year_month')['customer_id'].nunique()

first_month = df['order_year_month'].min()
last_month = df['order_year_month'].max()

first_month_customers = set(df[df['order_year_month'] == first_month]['customer_id'])
last_month_customers = set(df[df['order_year_month'] == last_month]['customer_id'])

new_customers = last_month_customers - first_month_customers
crr = ((len(last_month_customers) - len(new_customers)) / len(first_month_customers)) * 100

orders_per_customer = df.groupby('customer_id')['order_id'].nunique()

repeat_customers = len(orders_per_customer[orders_per_customer > 1])
total_customers = len(orders_per_customer)

rpr = (repeat_customers / total_customers) * 100


df['delivery_time_days'] = (
    df['order_delivered_customer_date'] - df['order_purchase_timestamp']
).dt.days

avg_fulfillment_time = df['delivery_time_days'].mean()

print("\n📊 --- KPI Summary ---")
print(f"Total Revenue: {total_revenue:,.2f}")
print(f"Average Order Value (AOV): {aov:,.2f}")
print(f"Customer Retention Rate (CRR): {crr:.2f}%")
print(f"Repeat Purchase Rate (RPR): {rpr:.2f}%")
print(f"Order Fulfillment Time: {avg_fulfillment_time:.2f} days")

kpi_df = pd.DataFrame({
    'total revenue': [total_revenue],
    'average order value': [aov],
    'delivery_time_days': [avg_fulfillment_time],
})

corr_matrix = kpi_df.corr()

print("\n🔹 Correlation Matrix:")
print(corr_matrix)

plt.figure(figsize=(12,6))
df.groupby('order_year_month')['price'].sum().plot(kind='bar', color='skyblue')
plt.title('Monthly Revenue Trend')
plt.xlabel('Year-Month')
plt.ylabel('Revenue')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
sns.histplot((df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days,
             bins=30, kde=True, color='lightgreen')
plt.title('Distribution of Delivery Duration')
plt.xlabel('Days to Deliver')
plt.ylabel('Count')
plt.show()