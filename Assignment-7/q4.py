import numpy as np
from scipy.stats import skew, kurtosis, norm
import matplotlib.pyplot as plt

data = [45, 67, 78, 89, 55, 90, 72, 60, 82, 76, 88, 92, 65, 70, 84, 79, 91, 74, 63,
        80, 55, 68, 77, 88, 59, 62, 95, 99, 85, 73, 60, 78, 81, 92, 66, 71, 83, 56,
        69, 87, 64, 75, 89, 90, 72, 61, 58, 93, 97, 100]

mean = np.mean(data)
variance = np.var(data, ddof=1)
skewness = skew(data)
kurt = kurtosis(data)

print("Mean:", mean)
print("Variance:", variance)
print("Skewness:", skewness)
print("Kurtosis:", kurt)

plt.hist(data, bins=10, density=True, alpha=0.6, color='g')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mean, np.std(data))
plt.plot(x, p, 'k', linewidth=2)
plt.show()
