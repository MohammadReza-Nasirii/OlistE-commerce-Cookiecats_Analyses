import pandas as pd

# --- Load datasets ---
customers = pd.read_csv('olist_customers_dataset.csv')
geolocation = pd.read_csv('olist_geolocation_dataset.csv')
order_items = pd.read_csv('olist_order_items_dataset.csv')
order_payments = pd.read_csv('olist_order_payments_dataset.csv')
order_reviews = pd.read_csv('olist_order_reviews_dataset.csv')
orders_data = pd.read_csv('olist_orders_dataset.csv')
products = pd.read_csv('olist_products_dataset.csv')
sellers = pd.read_csv('olist_sellers_dataset.csv')
products_category = pd.read_csv('product_category_name_translation.csv')

# --- Put them in a dictionary ---
datasets = {
    "Customers": customers,
    "geolocation": geolocation,
    "Order Items": order_items,
    "Order Payments": order_payments,
    "Order Reviews": order_reviews,
    "Orders": orders_data,
    "Products": products,
    "Sellers": sellers,
    "Product Category": products_category
}

# --- Loop through datasets ---
for name, df in datasets.items():
    print(f"\n{'='*80}")
    print(f"📂 Dataset: {name}")
    print(f"Shape: {df.shape}")
    print(f"\n🔹 Head:")
    print(df.head(3))
    print(f"\n🔹 Info:")
    df.info()