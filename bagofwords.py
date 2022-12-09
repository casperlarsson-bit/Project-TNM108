print(chr(27) + "[2J")

import pandas as pd
import nltk 
import numpy as np
import re
from nltk.stem import wordnet # to perform lemmitization
from sklearn.feature_extraction.text import CountVectorizer # to perform bow
from sklearn.feature_extraction.text import TfidfVectorizer # to perform tfidf
from nltk import pos_tag # for parts of speech
from sklearn.metrics import pairwise_distances # to perfrom cosine similarity
from nltk import word_tokenize # to create tokens
from nltk.corpus import stopwords # for stop words


from algorithms import *

# Read the dataset of answers
df = pd.read_excel('dataset.xlsx', engine='openpyxl')
df.ffill(axis=0, inplace=True) # Replace NaN answwers with the one above


# applying the fuction to the dataset to get clean text
df['lemmatized_text']=df['Questions'].apply(text_normalization) 

# all the stop words we have 
stop = stopwords.words('english')

cv = CountVectorizer() # intializing the count vectorizer
X = cv.fit_transform(df['lemmatized_text']).toarray()

# returns all the unique word from data 

features = cv.get_feature_names_out()
df_bow = pd.DataFrame(X, columns = features)

continueQuestions = True
while continueQuestions:
    Question = input() 

    if Question == 'bye' or Question == 'goodbye':
        continueQuestions = False

    # checking for stop words
    Q=[]
    a=Question.split()
    for i in a:
        if i in stop:
            continue
        else:
            Q.append(i)
        b=" ".join(Q) 
    

    index_value = bagOfWords(Question, df['lemmatized_text'])

    print('\nQuestion: ', Question)
    print('\nAnswer: ', df['Answers'].loc[index_value]) # The text at the above index becomes the response for the question)

    