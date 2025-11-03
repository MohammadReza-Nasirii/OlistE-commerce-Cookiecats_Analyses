import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

# Load datasets
orders = pd.read_csv('../Data/olist_orders_dataset.csv')
order_items = pd.read_csv('../Data/olist_order_items_dataset.csv')
products = pd.read_csv('../Data/olist_products_dataset.csv')
customers = pd.read_csv('../Data/olist_customers_dataset.csv')

# Convert date columns
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])

print("✅ Data Loaded Successfully!")
print(orders.head(2))
