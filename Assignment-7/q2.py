import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Automobile_data.csv")

print("Original Data:")
print(df.head())

num_cols = ["wheel-base", "length", "horsepower", "average-mileage", "price"]

scaler = StandardScaler()

df[num_cols] = scaler.fit_transform(df[num_cols])

print("\nStandardized Data (mean=0, std=1):")
print(df.head())
