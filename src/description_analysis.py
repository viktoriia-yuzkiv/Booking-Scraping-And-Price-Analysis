import pandas as pd
import numpy as np
import csv

import os
import re
import nltk
import spacy
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer 
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


from tqdm import tqdm
tqdm.pandas()

class Description_Analysis:
    
    def __init__(self):
        '''
        Constructor for the class
        '''
        sp = spacy.load('en_core_web_sm')

    def get_matrix_vectorizer(self, list_descriptions:list, ngram_range=(1,1)):
        '''
        Returns the matrix and vectorizer for the given column
        '''
        vectorizer = CountVectorizer(ngram_range)
        vectorizer.fit(list_descriptions)
        matrix = vectorizer.transform(list_descriptions)
        features = matrix.get_feature_names_out()
        return matrix, features
    
    def get_matrix_tfidf(self, list_descriptions:list, ngram_range=(1,1)):
        '''
        Returns the matrix and vectorizer for the given column
        '''
        vectorizer = TfidfVectorizer(ngram_range)
        vectorizer.fit(list_descriptions)
        matrix = vectorizer.transform(list_descriptions)
        features = vectorizer.get_feature_names()
        return matrix, features

    def get_stopwords(self):
        '''
        Returns the list of stopwords
        '''
        return stopwords.words('english')
    
    def get_lemmatizer(self):
        '''
        Returns the lemmatizer object
        '''
        return WordNetLemmatizer()