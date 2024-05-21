# expedia-click-hotel-recommendation
At this project I was given an Expedia log dataset which contained a 
sequence of hotel ID clicks for each client. A click is defined as a visit on a property 
description page that can originate from lodging sort, search engine (e.g. Google) 
or from lodging meta services (e.g. Trivago). The goal of the project was to predict 
the next click (hotel ID) a client would choose based on the previous clicks.
# Introduction
In order to achieve the goal, a Word2vec algorithm from the gensim python 
library was used. Word2vec is a neural network which takes batches of raw textual 
data and then transforms them to vector representations in a space of several 
dimensions. In our case, it would take the hotel click IDs as a word in a “sentence” 
context , as we suppose each client looks up for similar hotels in each online 
session on company’s website.
For that reason, the model was trained multiple times (by testing various 
values on 4 parameters) on a sample of 10% of the initially given train data, and 
tested on a validation data set using the Hits@k evaluation method.
# Steps
  1. We download the data from https://github.com/ExpediaGroup/pkdd22-challenge-expediagroup
  2. We take a 10% of train data (data preprocess)
  3. Then we further split this fraction in a 90/10 train validation data
  4. We split the clicks series into clicks/last click column (validation split)
  5. We prepare evaluation metric class (hits@K) and the model utils ( a function to find similar results of the click given)
  6. We run the model with a combination of different parameters
  7. Save the results in a txt file
  8. Plot the results and have a report file
