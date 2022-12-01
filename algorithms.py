import pandas as pd
import nltk
import numpy as np
import re
from nltk.stem import wordnet # to perform Lemmitization
from sklearn.feature_extraction.text import CountVectorizer # to perform bow
from sklearn.feature_extraction.text import TfidfVectorizer # to perform tfidf
from nltk import pos_tag # for parts of speech
from sklearn.metrics import pairwise_distances # to perfrom cosine similarity
from nltk import word_tokenize # to create tokens
from nltk.corpus import stopwords # for stop words

def text_normalization(text):
    text=str(text).lower() # text to lower case
    spl_char_text=re.sub(r'[^ a-z]','',text) # removing special characters
    tokens=nltk.word_tokenize(spl_char_text) # word tokenizing
    lema=wordnet.WordNetLemmatizer() # intializing lemmatization
    tags_list=pos_tag(tokens,tagset=None) # parts of speech
    lema_words=[]   # empty list 
    for token,pos_token in tags_list:
        if pos_token.startswith('V'):  # Verb
            pos_val='v'
        elif pos_token.startswith('J'): # Adjective
            pos_val='a'
        elif pos_token.startswith('R'): # Adverb
            pos_val='r'
        else:
            pos_val='n' # Noun
        lema_token=lema.lemmatize(token,pos_val) # performing lemmatization
        lema_words.append(lema_token) # appending the lemmatized token into a list
    
    return " ".join(lema_words) # returns the lemmatized tokens as a sentence 


def chat_tfidf(text, df):
    lemma = text_normalization(text)
    tfidf=TfidfVectorizer()
    x_tfidf=tfidf.fit_transform(df).toarray()
    dataFile_tfidf=pd.DataFrame(x_tfidf, columns=tfidf.get_feature_names_out())
    tf = tfidf.transform([lemma]).toarray()
    cos = 1- pairwise_distances(dataFile_tfidf, tf, metric='cosine')
    index_value1 = cos.argmax()
    return index_value1


def bagOfWords(text, df):
    lemma = text_normalization(text)
    cv = CountVectorizer()
    x = cv.fit_transform(df).toarray()
    features = cv.get_feature_names_out()
    dataFile_bow = pd.DataFrame(x, columns = features)
    qb = cv.transform([lemma]).toarray() # applying bow
    cos = 1-pairwise_distances(dataFile_bow, qb, metric = 'cosine')
    index_value = cos.argmax()
    return index_value

