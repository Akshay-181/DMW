import numpy as np

data = [45, 56, 67, 89, 90, 72, 60, 82, 76, 88, 92, 65, 70, 84, 79, 91, 74, 63, 80, 55]

p25 = np.percentile(data, 25)
p50 = np.percentile(data, 50)
p75 = np.percentile(data, 75)
p90 = np.percentile(data, 90)

print("25th Percentile:", p25)
print("50th Percentile (Median):", p50)
print("75th Percentile:", p75)
print("90th Percentile:", p90)
