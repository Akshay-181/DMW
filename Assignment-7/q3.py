import statistics as stats

marks = [45, 67, 78, 45, 90, 82, 67, 55, 45, 92, 88, 67]

mean = stats.mean(marks)
median = stats.median(marks)
mode = stats.mode(marks)
variance = stats.variance(marks)
std_dev = stats.stdev(marks)

print("Mean:", mean)
print("Median:", median)
print("Mode:", mode)
print("Variance:", variance)
print("Standard Deviation:", std_dev)
