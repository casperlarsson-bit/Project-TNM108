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

from algorithms import *
from textProcessing import *

stop = stopwords.words('english')

### Clean up the text from unwanted chars and make lower case
def text_normalizer(text):
    text = str(text).lower() #text to lower case
    spl_char_text = re.sub(r'[^a-z0-9]',' ', text) # removing special characters
    tokens = nltk.word_tokenize(spl_char_text) # word tokenizing, separating words
    lema = wordnet.WordNetLemmatizer() # initzializing lemmatization, grouping words with the same meaning
    tags_list = pos_tag(tokens, tagset=None) # parts of speech (ordklasser)
    lema_words = [] # empty list
    for token, pos_token in tags_list: # pos = parts of speech
        if pos_token.startswith('V'): # Verb
            pos_val = 'v'
        elif pos_token.startswith('J'): # Adjective
            pos_val = 'a'
        elif pos_token.startswith('R'): # Adverb
            pos_val = 'r'
        else:
            pos_val= 'n' # Noun
        lema_token= lema.lemmatize(token, pos_val) # performing lemmatization
        lema_words.append(lema_token) # appending the lemmatized token into a list

    return ' '.join(lema_words) # returns the lemmatized tokens as a sentence
    
def lemma(Question):
    Q = []
    a = Question.split()
    for i in a:
        if i in stop:
            continue
        else:
            Q.append(i)
        b = " ".join(Q)
        Question_lemma = text_normalizer(b)
    return Question_lemma