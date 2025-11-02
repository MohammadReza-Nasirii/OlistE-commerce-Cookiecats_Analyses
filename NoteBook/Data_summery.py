import pandas as pd

# Define your datasets in a dictionary
datasets = {
    "Customers": pd.read_csv('../Data/raw/olist_customers_dataset.csv'),
    "Orders": pd.read_csv('../Data/raw/olist_orders_dataset.csv'),
    "Order Items": pd.read_csv('../Data/raw/olist_order_items_dataset.csv'),
    "Payments": pd.read_csv('../Data/raw/olist_order_payments_dataset.csv'),
    "Reviews": pd.read_csv('../Data/raw/olist_order_reviews_dataset.csv'),
    "Products": pd.read_csv('../Data/raw/olist_products_dataset.csv'),
    "Sellers": pd.read_csv('../Data/raw/olist_sellers_dataset.csv')
}

# Save summaries to a text file
with open("data_summary.txt", "w", encoding="utf-8") as f:
    for name, df in datasets.items():
        f.write(f"===== {name} Dataset =====\n")
        f.write(f"Shape: {df.shape}\n\n")
        f.write("Columns:\n")
        f.write(", ".join(df.columns) + "\n\n")
        f.write("Head:\n")
        f.write(str(df.head(3)) + "\n\n")
        f.write("Info:\n")
        df.info(buf=f)
        f.write("\n\n" + "="*60 + "\n\n")
