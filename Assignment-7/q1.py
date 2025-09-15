import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("Automobile_data.csv")

print("Original Data:")
print(df.head())

num_cols = ["wheel-base", "length", "horsepower", "average-mileage", "price"]

scaler = MinMaxScaler()

df[num_cols] = scaler.fit_transform(df[num_cols])

print("\nRescaled Data (0 to 1):")
print(df.head())
