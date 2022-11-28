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
from main import dataFile, Question, algo

def train_bot():
    while True:
      print('\n(A) TFIDF: ' + dataFile['Answers'].loc[chat_tfidf(Question,dataFile['lemmatized_text'])])

      print('(B) BOW: ' + dataFile['Answers'].loc[bagOfWords(Question,dataFile['lemmatized_text'])])

      if Question =="Bye" or Question =="bye" or Question =="goodbye" or Question =="Goodbye":
        print('Goodbye')
        break
      else: 
         best_algo= input('\nBest algorithm? A / B / Equal\n Answer:')  
         if best_algo =='A' or best_algo =='a': 
            print('You chose ' + algo[0])
            
         elif best_algo =='B' or best_algo =='b': 
            print('You chose ' + algo[1])
            
         elif best_algo =='Equal' or best_algo =='equal': 
            print('The algorithms were equal')  
            
          #elif not best_algo=='A' or best_algo =='a' or best_algo =='B' or best_algo =='b' or best_algo =='Equal' or best_algo =='equal': 
         else:
            rate_algo(best_algo)

def rate_algo(best_algo):

    while True:
     try:
       if not best_algo=='A' or best_algo =='a':
          best_algo= input('Please select one option. \n Answer:') 
     except:
        continue   