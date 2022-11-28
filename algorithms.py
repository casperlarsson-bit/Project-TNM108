import pandas as pd
import nltk
import numpy as np
import re
from nltk.stem import wordnet #to perform lemmitization
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import  pos_tag #for parts of speech
from sklearn.metrics import pairwise_distances # to perform cosine similarity
from nltk import word_tokenize #to create tokens
from nltk.corpus import stopwords #for stop words

from textProcessing import *

def chat_tfidf(text, df):
    lemma = text_normalizer(text)
    tfidf=TfidfVectorizer()
    x_tfidf=tfidf.fit_transform(df).toarray()
    dataFile_tfidf=pd.DataFrame(x_tfidf, columns=tfidf.get_feature_names_out())
    tf = tfidf.transform([lemma]).toarray()
    cos = 1- pairwise_distances(dataFile_tfidf, tf, metric='cosine')
    index_value1 = cos.argmax()
    return index_value1

### bag of words (vectorizing the occurence of words in the document)
def bagOfWords(text, df):
    lemma = text_normalizer(text)
    cv = CountVectorizer()
    x = cv.fit_transform(df).toarray()
    features = cv.get_feature_names_out()
    dataFile_bow = pd.DataFrame(x, columns = features)
    qb = cv.transform([lemma]).toarray() # applying bow
    cos = 1-pairwise_distances(dataFile_bow, qb, metric = 'cosine')
    index_value2 = cos.argmax()
    return index_value2

