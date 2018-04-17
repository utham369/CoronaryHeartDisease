# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 09:38:05 2017

@author: Admin
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import math
import time
import re
import os
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity  
from sklearn.metrics import pairwise_distances
from matplotlib import gridspec
from scipy.sparse import hstack
import plotly
import plotly.figure_factory as ff
from plotly.graph_objs import Scatter, Layout
from sklearn.preprocessing import Imputer

plotly.offline.init_notebook_mode(connected=True)
warnings.filterwarnings("ignore")


# load a file

data = pd.read_excel('Data.xlsx')
data = data.rename(columns = {'id':'content_id' , 'content_name': 'title' ,'LONG_DESC':'description' ,'LANGUAGE': 'language'})

data.dtypes
data['content_name'] = data['title']
#data['title'] = data['title'].astype(str) + " "+ data['description']

data = data[['content_id','content_name' , 'title' ,'description','language','Genre','year']]


# consider products which have title information
#data = data.loc[~data['title'].isnull()]
data = data.loc[data['description'] != 'Coming Soon']
data = data.loc[~data['description'].isnull()]

print('Number of data points After title =NULL :', data.shape[0])

datattypes = data.dtypes

data = data.loc[~data['Genre'].isnull()]
#data_1 =data.loc[data['content_id']== 9887]
data = data.loc[~data['language'].isnull()]
data = data.loc[~data['year'].isnull()]
#data['year'] = data['year'].astype(str)
datattypes = data.dtypes


data['title'] = data['title'].astype(str) + " "+ data['description']
data = data[['content_id','content_name' , 'title','language','Genre','year']]


data.to_pickle('moviepickels/3k_latest_movie_data')
data.shape

data = pd.read_pickle('moviepickels/3k_latest_movie_data')


# Remove All products with very few words in title
data_sorted = data[data['title'].apply(lambda x: len(x.split())>4)]
print("After removal of products with short description:", data_sorted.shape[0])

data = data.reset_index(drop=True)
kick_movie = data.loc[data['content_id'] == 9100]

# we use the list of stop words that are downloaded from nltk lib.
stop_words = set(stopwords.words('english'))
print ('list of stop words:', stop_words)

def nlp_preprocessing(total_text, index, column):
    if type(total_text) is not int:
        string = ""
        for words in total_text.split():
            # remove the special chars in review like '"#$@!%^&*()_+-~?>< etc.
            word = ("".join(e for e in words if e.isalnum() or e == ','))
            # Conver all letters to lower-case
            word = word.lower()
            # stop-word removal
            if not word in stop_words:
                string += word + " "
        data[column][index] = string
    
start_time = time.clock()
# we take each title and we text-preprocess it.
for index, row in data.iterrows():
    nlp_preprocessing(row['title'], index, 'title')
# we print the time it took to preprocess whole titles 
print(time.clock() - start_time, "seconds")

data = data.set_index('content_id')

tfidf_title_vectorizer = TfidfVectorizer(min_df = 0)
tfidf_title_features = tfidf_title_vectorizer.fit_transform(data['title'])
print(tfidf_title_features)

def tfidf_model(doc_id, num_results):
    # doc_id: apparel's id in given corpus
    
    # pairwise_dist will store the distance from given input apparel to all remaining apparels
    # the metric we used here is cosine, the coside distance is mesured as K(X, Y) = <X, Y> / (||X||*||Y||)
    # http://scikit-learn.org/stable/modules/metrics.html#cosine-similarity
    pairwise_dist = pairwise_distances(tfidf_title_features,tfidf_title_features[doc_id])
    
    print ("FEATURES " , tfidf_title_features[doc_id])

    # np.argsort will return indices of 9 smallest distances
    indices = np.argsort(pairwise_dist.flatten())[0:num_results]
    #pdists will store the 9 smallest distances
    pdists  = np.sort(pairwise_dist.flatten())[0:num_results]

    #data frame indices of the 9 smallest distace's
    df_indices = list(data.index[indices])

    #print (df_indices[1])
    for i in range(0,len(indices)):
        # we will pass 1. doc_id, 2. title1, 3. title2, url, model
   #     get_result(indices[i], data['title'].loc[df_indices[0]], data['title'].loc[df_indices[i]], data['medium_image_url'].loc[df_indices[i]], 'tfidf')
         print('content_id       :',data['content_id'].loc[df_indices[i]])
         print('content_name     :',data['content_name'].loc[df_indices[i]])
         print('language         :',data['language'].loc[df_indices[i]])
         
         print('Genre            :',data['Genre'].loc[df_indices[i]])
        
         print('Year            :',data['year'].loc[df_indices[i]])
         
         print ('Eucliden distance from the given image :', pdists[i])
 #       print('='*125)

# in the output heat map each value represents the tfidf values of the label word, the color represents the intersection with inputs title
 
temper_movie = data.loc[data['content_name'] == 'Darling']
tfidf_model(551, 10)


language_vectorizer = CountVectorizer()
language_features   = language_vectorizer.fit_transform(data['language'])

genre_vectorizer = CountVectorizer()
genre_features = genre_vectorizer.fit_transform(data['Genre'])

#sdfsf = genre_features

#print (genre_features)
#year_vectorizer = CountVectorizer()
#year_features = genre_vectorizer.fit_transform(data['year'])




extra_features = hstack((language_features, genre_features)).tocsr()


def idf_w2v_brand(doc_id, w1, w2, num_results):
    # doc_id: apparel's id in given corpus
    # w1: weight for  w2v features
    # w2: weight for brand and color features

    # pairwise_dist will store the distance from given input apparel to all remaining apparels
    # the metric we used here is cosine, the coside distance is mesured as K(X, Y) = <X, Y> / (||X||*||Y||)
    # http://scikit-learn.org/stable/modules/metrics.html#cosine-similarity
    idf_w2v_dist  = pairwise_distances(tfidf_title_features,tfidf_title_features[doc_id])
    
    #idf_w2v_dist  = pairwise_distances(idf_title_features,idf_title_features[doc_id])
    ex_feat_dist = pairwise_distances(extra_features, extra_features[doc_id])
    pairwise_dist   = (w1 * idf_w2v_dist +  w2 * ex_feat_dist)/float(w1 + w2)

    # np.argsort will return indices of 9 smallest distances
    indices = np.argsort(pairwise_dist.flatten())[0:num_results]
    #pdists will store the 9 smallest distances
    pdists  = np.sort(pairwise_dist.flatten())[0:num_results]

    #data frame indices of the 9 smallest distace's
    df_indices = list(data.index[indices])
    

    for i in range(0,len(indices)):
        # we will pass 1. doc_id, 2. title1, 3. title2, url, model
   #     get_result(indices[i], data['title'].loc[df_indices[0]], data['title'].loc[df_indices[i]], data['medium_image_url'].loc[df_indices[i]], 'tfidf')
         print('content_id       :',data['content_id'].loc[df_indices[i]])
         print('content_name     :',data['content_name'].loc[df_indices[i]])
         print('language         :',data['language'].loc[df_indices[i]])
         
         print('Genre            :',data['Genre'].loc[df_indices[i]])
        
         print('Year            :',data['year'].loc[df_indices[i]])
         
         print ('Eucliden distance from the given image :', pdists[i])
         
    return df_indices
 
    
df_indices =idf_w2v_brand(1565, 5, 15, 50)



def getMovieByYear(df_indices):
    yearwisedata = []
    data['year'] = data['year'].astype(np.int64)
    yearOfRelease = data['year'].loc[df_indices[0]]
    print ('YearOfRelease ' , yearOfRelease)
    currentYear = pd.tslib.Timestamp.now()
    year = currentYear.year
    diff_year =year - yearOfRelease
    print ('year ' , diff_year)
    if(diff_year <= 5 or diff_year == currentYear):
        print ("IF BLOCK")
        status= 'new_movie'
        countValue = 2
    else:
        print ("else BLOCasdK")
        status= 'old_movie'
        countValue = -2
    
    for x in range (10):
        print ("IF BLOasdaaCK")
        if(status == 'new_movie'):
            #print ("IF BLOasdaaCK")
            rel_year = yearOfRelease + countValue
            yearwisedata.append(rel_year)
            countValue = countValue - 1
            print ("MOVIES BY YEAR " ,countValue , rel_year)  
        else:
            rel_year = yearOfRelease + countValue
            yearwisedata.append(rel_year)
            countValue = countValue + 1
            print ("MOVIES BY YEAR " ,countValue , rel_year) 
            print (yearwisedata)  
            yearwisedata.sort()
    return yearwisedata


mvie_year_data = getMovieByYear(df_indices)

print ('MOVIE  : ' , mvie_year_data)
print (df_indices)

#print (data['year'].loc[df_indices[0])
print('Year            :',data['year'].loc[df_indices[0]])
print('Year            :',df_indices[0])



def processMovies(df_indices , beginYear):
    new_df_indices= []
    df_indices.sort()
    for x in range(0 , len(df_indices)):
        if(data['year'].loc[df_indices[x]] > beginYear):
            new_df_indices.append(df_indices[x])
            
    return new_df_indices

new_df_indices = processMovies(df_indices,mvie_year_data[0])

print (new_df_indices)

def displayMovies(new_df_indices):
    for i in range(0,len(new_df_indices)):
        # we will pass 1. doc_id, 2. title1, 3. title2, url, model
   #     get_result(indices[i], data['title'].loc[df_indices[0]], data['title'].loc[df_indices[i]], data['medium_image_url'].loc[df_indices[i]], 'tfidf')
         print('content_id       :',data['content_id'].loc[new_df_indices[i]])
         print('content_name     :',data['content_name'].loc[new_df_indices[i]])
         print('language         :',data['language'].loc[new_df_indices[i]])
         
         print('Genre            :',data['Genre'].loc[new_df_indices[i]])
        
         print('Year            :',data['year'].loc[new_df_indices[i]])
    
displayMovies(new_df_indices)




                
            
            

             
            
       
            
        

for x in df_indices:
    print (x)
    
#idf_w2v_brand(357, 5, 5, 10)
#idf_w2v_brand(3346, 5, 5, 10)
#tfidf_model(1577, 10)

 



