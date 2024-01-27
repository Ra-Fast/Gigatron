import numpy as np
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_regression
import pandas as pd

def calculate_entropy(data, bins):
    entropy_values = []
    for column in data.columns:
        hist, bin_edges = np.histogram(data[column], bins=bins, range=(0, 1), density=True)
        entropy_values.append(entropy(hist, base=2))
    return entropy_values

def calculate_mutual_information(data, bins):
    mutual_info_values = []
    for i in range(len(data.columns)):
        for j in range(i + 1, len(data.columns)):
            mi = mutual_info_regression(data.iloc[:, i:i+1], data.iloc[:, j])
            mutual_info_values.append(mi)
    return mutual_info_values

def calculate_relative_entropy(data1, data2, bins):
    relative_entropy_values = []
    for i in range(len(data1.columns)):
        kl_distance = entropy(data1.iloc[:, i], qk=data2.iloc[:, i], base=2)
        relative_entropy_values.append(kl_distance)
    return relative_entropy_values

# Example usage
# Assuming df1 and df2 are your two DataFrames with continuous values between 0 and 1
# You can replace these with your actual DataFrames
data1 = pd.DataFrame({
    'col1': np.random.rand(1000),
    'col2': np.random.rand(1000),
    'col3': np.random.rand(1000)
})

data2 = pd.DataFrame({
    'col1': np.random.rand(1000),
    'col2': np.random.rand(1000),
    'col3': np.random.rand(1000)
})

bins = 10  # You can change the number of bins as needed

entropy_values = calculate_entropy(data1, bins)
mutual_info_values = calculate_mutual_information(data1, bins)
relative_entropy_values = calculate_relative_entropy(data1, data2, bins)

print("Entropy values for each column:", entropy_values)
print("Mutual information values between columns:", mutual_info_values)
print("Relative entropy values between datasets:", relative_entropy_values)
