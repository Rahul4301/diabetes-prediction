import pandas as pd
import numpy as np

# Load the existing dataset
df = pd.read_csv('diabetes1.csv')

# Function to generate synthetic data
def generate_synthetic_data(df, num_samples):
    synthetic_data = []
    for _ in range(num_samples):
        sample = df.sample(n=1).values[0]
        noise = np.random.normal(0, 0.1, sample.shape)
        synthetic_sample = sample + noise
        synthetic_data.append(synthetic_sample)
    return np.array(synthetic_data)

# Generate 50,000 synthetic samples
num_samples = 50000
synthetic_data = generate_synthetic_data(df, num_samples)

# Introduce some outliers
num_outliers = int(0.01 * num_samples)  # 1% outliers
outliers = np.random.uniform(low=-10, high=10, size=(num_outliers, df.shape[1]))
synthetic_data[:num_outliers] = outliers

# Combine original and synthetic data
combined_data = np.vstack([df.values, synthetic_data])

# Save the new dataset to a CSV file
new_df = pd.DataFrame(combined_data, columns=df.columns)
new_df.to_csv('/Users/rahulsuthar/Documents/projects/remove/diabetes/diabetes_extended.csv', index=False)

print("New dataset with 50,000 synthetic entries created successfully.")