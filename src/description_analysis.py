import pandas as pd
import numpy as np
import csv

import os
import re
import nltk
import spacy
import statsmodels.api as sm
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer 
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

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
        self.sp = spacy.load('en_core_web_sm')
        self.stop_words = set(stopwords.words('english'))

    # Parameters: 
    # 1. list_descriptions: list of descriptions
    # 2. Snowball: boolean to use SnowballStemmer or PorterStemmer (default is True)
    def clean_text_stemming(self, list_descriptions:list, Snowball = True):
        '''
        Cleans the text and returns the stemmed text
        '''
        if Snowball:
            stemmer = SnowballStemmer("english")
        else:  
            stemmer = PorterStemmer("english")
        
        cleaned_text = []
        for description in list_descriptions:
            description = re.sub(r'[^\w\s]', '', description)
            description = re.sub(r'\d+', '', description)
            description = [word if re.match('([A-Z]+[a-z]*){2,}', description) else description.lower() for word in description.split()]
            description = [word for word in description if word not in self.stop_words]
            description = [stemmer.stem(word) for word in description]
            description = ','.join(description)
            cleaned_text.append(description)
        return cleaned_text
    
    def clean_text_lemmatization(self, list_descriptions:list):
        '''
        Cleans the text and returns the lemmatized text
        '''
        lemmatizer = WordNetLemmatizer()
        cleaned_text = []
        for description in list_descriptions:
            description = re.sub(r'[^\w\s]', '', description)
            description = re.sub(r'\d+', '', description)
            description = [word if re.match('([A-Z]+[a-z]*){2,}', description) else description.lower() for word in description.split()]
            description = [word for word in description if word not in self.stop_words]
            description = [lemmatizer.stem(word) for word in description]
            description = ','.join(description)
            cleaned_text.append(description)
        return cleaned_text

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
    
    #additional functions
    def strip(self, word):
        mod_string = re.sub(r'\W+', '', word)
        return mod_string

    #the following leaves in place two or more capital letters in a row
    #will be ignored when using standard stemming
    def abbr_or_lower(self, word):
        if re.match('([A-Z]+[a-z]*){2,}', word):
            return word
        else:
            return word.lower()

    #modular pipeline for stemming, lemmatizing and lowercasing
    #note this is NOT lemmatizing using grammar pos
    def tokenize(self, text, modulation=1):

        # TODO: Adapt to Stemmer
        porter = PorterStemmer()

        if modulation<2:
            tokens = re.split(r'\W+', text)
            stems = []
            # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
            for token in tokens:
                lowers=self.abbr_or_lower(token)
                if lowers not in self.stop_words:
                    if re.search('[a-zA-Z]', lowers):
                        if modulation==0:
                            stems.append(lowers)
                        if modulation==1:
                            stems.append(porter.stem(lowers))
        else:
            sp_text=sp(text)
            stems = []
            lemmatized_text=[]
            for word in sp_text:
                lemmatized_text.append(word.lemma_)
            stems = [self.abbr_or_lower(self.strip(w)) for w in lemmatized_text if (self.abbr_or_lower(self.strip(w))) and (self.abbr_or_lower(self.strip(w)) not in self.stop_words)]
        return " ".join(stems)


    def vectorize(self, tokens, vocab):
        vector=[]
        for w in vocab:
            vector.append(tokens.count(w))
        return vector
    
    def get_best_coefficients(self, df, TFidf = False, n_gram_min = 1, n_gram_max = 1):
        # Step 0: No, nan, and empty string removal
        df = df[df['Hotel_Description_Long'].notna()]

        # Step 1: Apply CountVectorizer
        if TFidf:
            # Apply TfidfVectorizer with limit 

            vectorizer = TfidfVectorizer(max_features=5000, ngram_range = (n_gram_min,n_gram_max), min_df=0.0, max_df=0.95)
        else:
            vectorizer = CountVectorizer(max_features=5000, ngram_range = (n_gram_min,n_gram_max), min_df=0.0, max_df=0.95)
        treatment_matrix = vectorizer.fit_transform(df['Hotel_Description_Long'])

        # Set column names to feature names
        treatment_df = pd.DataFrame(treatment_matrix.toarray(), columns=vectorizer.get_feature_names_out())
        
        # Step 2: Concatenate matrices with price columns
        treatment_concatenated = pd.concat([treatment_df, df['Price']], axis=1)

        # Step 3: Identify common words
        common_words = list(set(vectorizer.get_feature_names_out()))
        print(f"Number of common words: {len(common_words)}")

        # Initialize variables to store the top 3 words and coefficients
        top_words_coefficients = []
        top_words_r_squared = []

        # Remove all nan
        treatment_concatenated = treatment_concatenated.dropna()

        # Iterate over each word in common_words
        for common_word in common_words:
            # Fit OLS model for treatment group
            X_treatment = sm.add_constant(treatment_concatenated[common_word])
            y_treatment = treatment_concatenated['Price']
            model_treatment = sm.OLS(y_treatment, X_treatment).fit()

            # Get coefficient value
            coefficient_treatment = model_treatment.params[common_word]

            # Check if this word should be included in the top 3
            if len(top_words_coefficients) < 3:
                top_words_coefficients.append((common_word, coefficient_treatment))
                # Sort the list based on coefficients in descending order
                top_words_coefficients.sort(key=lambda x: x[1], reverse=True)
            else:
                # Check if the current word has a higher coefficient than the smallest coefficient in the top 3
                if coefficient_treatment > top_words_coefficients[-1][1]:
                    top_words_coefficients[-1] = (common_word, coefficient_treatment)
                    # Sort the list based on coefficients in descending order
                    top_words_coefficients.sort(key=lambda x: x[1], reverse=True)
            if len(top_words_r_squared) < 3:
                top_words_r_squared.append((common_word, model_treatment.rsquared))
                # Sort the list based on coefficients in descending order
                top_words_r_squared.sort(key=lambda x: x[1], reverse=True)
            else:
                # Check if the current word has a higher coefficient than the smallest coefficient in the top 3
                if model_treatment.rsquared > top_words_r_squared[-1][1]:
                    top_words_r_squared[-1] = (common_word, model_treatment.rsquared)
                    # Sort the list based on coefficients in descending order
                    top_words_r_squared.sort(key=lambda x: x[1], reverse=True)

        # Print the top 3 words and their coefficients
        for i, (word, coefficient) in enumerate(top_words_coefficients, start=1):
            print(f"Top {i} word based on coefficient: {word} ({coefficient})")
        for i, (word, r_squared) in enumerate(top_words_r_squared, start=1):
            print(f"Top {i} word based on R-squared: {word} ({r_squared})")

        print("")
        return top_words_coefficients, model_treatment.summary()


    def get_selected_features(self, df, selected_features: list, TFidf = False):
        # Make sure selected features is a list of strings max 2
        assert len(selected_features) <= 3, "Selected features should be a list of strings with max 2 elements"
        assert all(isinstance(feature, str) for feature in selected_features), "All elements in selected features should be strings"
        
        n_gram_min = 1
        n_gram_max = 1
        for feature in selected_features:
            n_gram = len(feature.split())
            if n_gram > n_gram_max:
                n_gram_max = n_gram
            if n_gram < n_gram_min:
                n_gram_min = n_gram

        # Step 0: No, nan, and empty string removal
        df = df[df['Hotel_Description_Long'].notna()]

        # Step 1: Apply CountVectorizer
        if TFidf:
            vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(n_gram_min,n_gram_max), min_df=0.0, max_df=0.95)
        else:
            vectorizer = CountVectorizer(max_features=5000, ngram_range=(n_gram_min,n_gram_max), min_df=0.0, max_df=0.95)
        treatment_matrix = vectorizer.fit_transform(df['Hotel_Description_Long'])
        
        # Set column names to feature names
        treatment_df = pd.DataFrame(treatment_matrix.toarray(), columns=vectorizer.get_feature_names_out())

        # Check if preprocessed selected features are present in the matrix columns
        present_features = [feature.lower() for feature in selected_features if feature.lower() in treatment_df.columns]

        if not present_features:
            raise ValueError("None of the selected features are present in the CountVectorizer matrix.")

        # Step 2: Concatenate matrices with price columns
        treatment_concatenated = pd.concat([treatment_df[present_features], df['Price']], axis=1)

        # Remove all nan
        treatment_concatenated = treatment_concatenated.dropna()

        # Step 3: Fit OLS model for treatment group
        X_treatment = sm.add_constant(treatment_concatenated[present_features])
        y_treatment = treatment_concatenated['Price']
        model_treatment = sm.OLS(y_treatment, X_treatment).fit()

        # Return the model summary
        return model_treatment, treatment_df

    def get_sentiment(self, df1, df2):
        # Remove nan
        df1 = df1[df1['Hotel_Description_Long'].notna()]
        df2 = df2[df2['Hotel_Description_Long'].notna()]

        vectorizer = CountVectorizer(ngram_range=(1,2), min_df=0.0, max_df=0.95)
        matrix1 = vectorizer.fit_transform(df1['Hotel_Description_Long'])
        matrix2 = vectorizer.transform(df2['Hotel_Description_Long'])

        sid = SentimentIntensityAnalyzer()

        # Add TF-IDF features to DataFrame
        tfidf_features = pd.DataFrame(matrix1.toarray(), columns=vectorizer.get_feature_names_out())
        copy_df1 = pd.concat([df1, tfidf_features], axis=1)
        copy_df2 = pd.concat([df2, tfidf_features], axis=1)

        # Sentiment Analysis
        sia = SentimentIntensityAnalyzer()

        # Create an empty Series to store sentiment scores
        sentiments = pd.Series(index=copy_df1.index)

        # Iterate over each hotel description and perform sentiment analysis
        for idx, description in enumerate(df1['Hotel_Description_Long']):
            if pd.notnull(description):  # Check for missing descriptions
                sentiments[idx] = sia.polarity_scores(description)['compound']
            else:
                sentiments[idx] = None

        # Add the sentiment Series to the DataFrame
        copy_df1['Sentiment'] = sentiments

        # Summary Statistics for Text Features
        summary_statistics = copy_df1.groupby(['City']).agg({
            'Sentiment': ['mean', 'std'],
            # Add additional features as needed
        }).reset_index()
        print(summary_statistics)