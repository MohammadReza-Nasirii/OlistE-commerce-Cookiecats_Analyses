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


