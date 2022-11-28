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
from menuOptions import *       

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

### Loop the data lines
# with open("dialogs.txt", 'r') as dataFile:
#     # get No of columns in each line
#     col_count = [ len(line.split("\t")) for line in dataFile.readlines() ]

### Generate column names  (names will be 0, 1, 2, ..., maximum columns - 1)
# column_names = ['Questions', 'Answers']

### Read csv
# dataFile = pd.read_csv("dialogs.txt", header=None, delimiter="\t", names=column_names) 

dataFile = pd.read_excel("Q_and_A.xlsx") 

dataFile.ffill(axis=0, inplace = True)

### Applying to dataset
dataFile['lemmatized_text'] = dataFile['Questions'].apply(text_normalizer)

menu = {}
menu['1']="Talk to the bot" 
menu['2']="Validate the bot"
menu['3']="Exit"

algo = ['TFIDF','BOW']

while True: 
    print('\n--------------------- ')
    options=menu.keys()
    sorted(options)
    for entry in options: 
        print (entry, menu[entry])
    selection = input("Please select: ") 
    print('--------------------- ')
  
    if selection =='1': 
      print ('\nLets chat!') 
      Question = input('Start talking: ')

      print('TFIDF: ' + dataFile['Answers'].loc[chat_tfidf(Question,dataFile['lemmatized_text'])])

      print('BOW: ' + dataFile['Answers'].loc[bagOfWords(Question,dataFile['lemmatized_text'])])
      
      if Question =="Bye" or Question =="bye" or Question =="goodbye" or Question =="Goodbye":
          print('Goodbye')
          break  

    elif selection == '2': 
     print ('\nLets train the set!')
     Question = input('Start talking: ') 
     train_bot()
    elif selection == '': 
      break
    elif selection == '3':
      break
    else: 
      print ('Unknown Option Selected')

    Question = input()

    print('TFIDF: ' + dataFile['Answers'].loc[chat_tfidf(Question,dataFile['lemmatized_text'])])

    print('BOW: ' + dataFile['Answers'].loc[bagOfWords(Question,dataFile['lemmatized_text'])])

