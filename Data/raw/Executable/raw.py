import kagglehub
import pandas as pd
import os

# Download datasets
path_olist = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
path_cookiecats = kagglehub.dataset_download("marwandiab/cookie-catsdataset")

print("Olist dataset path:", path_olist)
print("Cookie Cats dataset path:", path_cookiecats)


