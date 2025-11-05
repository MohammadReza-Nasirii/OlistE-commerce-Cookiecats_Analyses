import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')

orders = pd.read_csv('../Data/raw/olist_orders_dataset.csv')
customers = pd.read_csv('../Data/raw/olist_customers_dataset.csv')

merged_df = pd.merge(orders, customers, on='customer_id', how='left')

merged_df['order_purchase_timestamp'] = pd.to_datetime(merged_df['order_purchase_timestamp'])
merged_df['order_approved_at'] = pd.to_datetime(merged_df['order_approved_at'])
merged_df['order_delivered_customer_date'] = pd.to_datetime(merged_df['order_delivered_customer_date'])

total_customers = merged_df['customer_unique_id'].nunique()

repeat_customers = merged_df['customer_unique_id'].value_counts()
repeat_customers_count = (repeat_customers > 1).sum()

repeat_customer_rate = (repeat_customers_count / total_customers) * 100

avg_orders_per_customer = merged_df['customer_unique_id'].value_counts().mean()

merged_df['order_year'] = merged_df['order_purchase_timestamp'].dt.year
retention_df = merged_df.groupby(['customer_unique_id', 'order_year']).size().unstack(fill_value=0)
retained_customers = (retention_df.sum(axis=1) > 1).sum()
retention_rate = (retained_customers / total_customers) * 100

print(f"Total Customers: {total_customers}")
print(f"Repeat Customers: {repeat_customers_count} ({repeat_customer_rate:.2f}%)")
print(f"Average Orders per Customer: {avg_orders_per_customer:.2f}")
print(f"Customer Retention Rate (CRR): {retention_rate:.2f}%")

labels = ['Repeat Customers', 'One-time Customers']
sizes = [repeat_customers_count, total_customers - repeat_customers_count]
colors = ['#4CAF50', '#FFC107']

plt.figure(figsize=(6,6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('Customer Purchase Behavior')
plt.show()

