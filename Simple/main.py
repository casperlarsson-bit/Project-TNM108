print(chr(27) + "[2J")

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

df = pd.read_excel('dataset.xlsx', engine='openpyxl')
#print(df.head())

df.ffill(axis=0, inplace=True) # Replace NaN answwers with the one above
#print(df.head(5))

df1 = df.head(10) # Copy first 10 elements of dataset

# Remove special characters and make to lower
def step1(x):
    for i in x:
        a = str(i).lower()
        p = re.sub(r'[^a-z0-9]',' ', a)
        #print(p)

step1(df1['Questions'])




#Word Tokenizing, Create a vector that separates each word in a sentence

s='tell me aobut your personality'

words = word_tokenize(s)
#print(words)

pos_tag(word_tokenize(s), tagset = None) # returns the parts of speech of every word



lemma = wordnet.WordNetLemmatizer()
lemma.lemmatize('absorbed', pos = 'v')

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

#print(text_normalization('telling you some stuff about me'))

df['lemmatized_text']=df['Questions'].apply(text_normalization) # applying the fuction to the dataset to get clean text
df.tail(15)

# all the stop words we have 
stop = stopwords.words('english')

cv = CountVectorizer() # intializing the count vectorizer
X = cv.fit_transform(df['lemmatized_text']).toarray()

# returns all the unique word from data 

features = cv.get_feature_names_out()
df_bow = pd.DataFrame(X, columns = features)
df_bow.head()

Question = input() #'Will you help me and tell me about yourself more' # considering an example query

# checking for stop words

Q=[]
a=Question.split()
for i in a:
    if i in stop:
        continue
    else:
        Q.append(i)
    b=" ".join(Q) 


Question_lemma = text_normalization(b) # applying the function that we created for text normalizing
Question_bow = cv.transform([Question_lemma]).toarray() # applying bow
#print(Question_bow)

# cosine similarity for the above question we considered.

cosine_value = 1- pairwise_distances(df_bow, Question_bow, metric = 'cosine' )
(cosine_value)

df['similarity_bow']=cosine_value # creating a new column 

df_simi = pd.DataFrame(df, columns=['Answers','similarity_bow']) # taking similarity value of responses for the question we took
#print(df_simi) 

df_simi_sort = df_simi.sort_values(by='similarity_bow', ascending=False) # sorting the values
#print(df_simi_sort.head())

threshold = 0.2 # considering the value of p=smiliarity to be greater than 0.2
df_threshold = df_simi_sort[df_simi_sort['similarity_bow'] > threshold] 
#print(df_threshold)

index_value = cosine_value.argmax() # returns the index number of highest value
print('\nQuestion: ', Question)
print('\nAnswer: ', df['Answers'].loc[index_value]) # The text at the above index becomes the response for the question)