import pandas as pd

file_path = 'C:///project ds/train_validation_sample/validation_sample.csv'
validation_data = pd.read_csv(file_path)

# Split the 'clicks' column into a list of clicks
validation_data['clicks_list'] = validation_data['clicks'].apply(lambda x: x.split(','))

# Extract the last click from each list and store it in a new column 'last_click'
validation_data['last_click'] = validation_data['clicks_list'].apply(lambda x: x[-1] if x else None)

# Rejoin the remaining clicks back into a comma-separated string, excluding the last click
validation_data['clicks'] = validation_data['clicks_list'].apply(lambda x: ','.join(x[:-1]) if x else None)

# Drop the temporary 'clicks_list' column 
validation_data.drop(columns=['clicks_list'], inplace=True)

# Export the validation_data DataFrame to a CSV file
validation_data.to_csv('C:///project ds/validation_data.csv', index=False)