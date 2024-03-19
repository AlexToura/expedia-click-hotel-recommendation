# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 12:10:56 2024

@author: gep
"""

import gensim
from gensim.models import Word2Vec
import pandas as pd
import random


train_df = pd.read_table('train.tsv', sep = '\t', chunksize = 100000)

df = pd.DataFrame()
for chunk in train_df:
    temp_df = chunk.dropna(axis = 0)
    temp_df = temp_df['clicks'].str.split(',', expand = True).iloc[:, 0:5].dropna(axis = 0)\
        .sample(frac = 0.10)
    if temp_df.isnull().values.any() == False:
        df = pd.concat([df, temp_df], axis = 0)
    else:
        pass

df['clicks'] = df[df.columns[0:]].apply(
    lambda x: ','.join(x.dropna().astype(str)), axis = 1)

target_click = df.iloc[:, 4]

sentences = df.iloc[:, :4]
sentences = sentences[sentences.columns[0:]].apply(
    lambda x: ','.join(x.dropna().astype(str)), axis = 1)

sentences = sentences.tolist()

# Train Word2Vec model
word2vec_model = Word2Vec(sentences, vector_size=1, window=4, min_count=5, workers=4)

# Function to convert clicks to embeddings
def clicks_to_embeddings(clicks, model):
    embeddings = []
    for click in clicks:
        try:
            embeddings.append(model.wv[click])
        except KeyError:
            embeddings.append([0] * model.vector_size)
    return embeddings

# Convert clicks to embeddings
embeddings_df = df['clicks'].apply(lambda x: clicks_to_embeddings(x, word2vec_model))
embeddings_df.to_csv('clicks_embeddings.csv', index = False)
df.head()

df.to_csv('df_clicks.csv', index = False)

df.drop([0, 1, 2, 3, 4 ], axis = 1, inplace = True)


df['click_embeddings'].to_csv('clicks_embeddings.csv', index = False)



















