import pandas as pd
from gensim.models import Word2Vec
import numpy as np
from model_utils import find_similar_clicks
from evaluation_metrics import HitsAtK

# Load the training data
csv_file_path = '//project ds/train_validation_sample/train_sample.csv'
train_sample = pd.read_csv(csv_file_path)
train_sample['clicks_list'] = train_sample['clicks'].apply(lambda x: x.split(','))
click_sequences = train_sample['clicks_list'].tolist()

# Load the validation data
validation_file_path = '//project ds/validation_data.csv'
validation_data = pd.read_csv(validation_file_path)
validation_data = validation_data.iloc[:100000]

# Parameter ranges
negative_sampling_values = [200]
vector_sizes = [32]
windows = [4]
min_counts = [1,2,5]

# Open a file to write the results
results_file = '//project ds/model_results.txt'
with open(results_file, 'w') as file:
    file.write("Model Parameters and Hits@k Scores\n")
    file.write("-----------------------------------\n")

    # Loop through each combination of parameters
    for ns in negative_sampling_values:
        for vec_size in vector_sizes:
            for win in windows:
                for min_count in min_counts:
                    # Train the Word2Vec model
                    model = Word2Vec(sentences=click_sequences, vector_size=vec_size, window=win,
                                     min_count=min_count, workers=4, negative=ns)

                    # Apply the model to find similar clicks for the validation data
                    validation_data['topN_similar_clicks'] = validation_data['clicks'].apply(
                        lambda x: find_similar_clicks(model, x))
                    validation_data['topN_similar_clicks'] = validation_data['topN_similar_clicks'].apply(
                        lambda x: [click[0] for click in x] if x else [])

                    # Calculate hits@k using the HitsAtK class
                    validation_data['last_click'] = validation_data['last_click'].astype(str)
                    hits_evaluator = HitsAtK(k=10)
                    for index, row in validation_data.iterrows():
                        hits_evaluator.update_metric(row['topN_similar_clicks'], row['last_click'])

                    hits_at_k_score = hits_evaluator.calculate_metric()
                    params = {'vector_size': vec_size, 'window': win, 'min_count': min_count, 'negative': ns}
                    result_text = f"Params: {params}, Hits@k Score: {hits_at_k_score:.5f}\n"
                    file.write(result_text)
                    print(f"Completed: {params} with score {hits_at_k_score:.5f}")
