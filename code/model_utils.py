
from gensim.models import Word2Vec

def find_similar_clicks(model, clicks_string, topn=11):
    # Split the clicks into a list and grab the last click
    clicks_list = clicks_string.split(',')
    last_click = clicks_list[-1]

    try:
        # Get the 11 most similar items (including the last click itself)
        most_similar = model.wv.most_similar(last_click, topn=topn)

        # Exclude the last click from the result and get the top 10
        filtered_similar = [item for item in most_similar if item[0] != last_click][:10]
        return filtered_similar
    except KeyError:
        # Return None if the last click is not in the model's vocabulary
        return  []
