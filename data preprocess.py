import pandas as pd


file_path = 'C://project ds/train.tsv'

# Read the TSV file into a pandas DataFrame
train_data = pd.read_csv(file_path, delimiter='\t')

# Display the first few rows of the DataFrame to understand its structure
print(train_data.head())

# Drop any rows with missing values
train_data.dropna(inplace=True)

# Sample 10% of the data, ensuring reproducibility with a random_state
sampled_data = train_data.sample(frac=0.1, random_state=42)

train_sample = sampled_data.sample(frac=0.9, random_state=42)
validation_sample = sampled_data.drop(train_sample.index)

# Export the train_sample DataFrame to a CSV file
train_sample.to_csv('C://project ds/train_sample.csv', index=False)

# Export the validation_sample DataFrame to a CSV file
validation_sample.to_csv('C:/project ds/validation_sample.csv', index=False)