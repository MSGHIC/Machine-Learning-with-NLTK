# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 21:23:31 2018

@author: MSG
"""
#Natural languange Understanding

#****SENTIMENT ANALYSIS WITH NLTK*****

#import required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the dataset
#quoting=3 , ignore quotes in text, delimiter ='\t' ... tab is used to separate text
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
#cleaning the text
#import library for cleaning
import re
#import library for removing non significant words
import nltk
nltk.download('stopwords')# contains list of irrelevant words
from nltk.corpus import stopwords
#class for steming(consider root words only)
from nltk.stem.porter import PorterStemmer
#clean list(corpus...collection of text)
corpus = []
for i in range(0, 1000):
    #[^a-zA-Z] means keep only characters a-z & A-Z
    #dataset['Reviews'][0] where?
    #' ' space replaces removed character
    review = re.sub('[^a-zA-Z]', ' ',  dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #convert review list back to single string
    review = ' '.join(review)
    corpus.append(review)
    
#*****CREATE BAG OF WORDS MODEL*******
    #takes each word(unique, no duplicates) from corpus and creates a column for it
    #hence creating a table with 1000 rows(each row for a review) 
    #each cell will contain a number representing the number of times the column word 
    #appears in the review
    
from sklearn.feature_extraction.text import CountVectorizer
#max_features = number of most frequent words
cv = CountVectorizer(max_features = 1500)
#create sparse matrix(matrix of independent features)
X = cv.fit_transform(corpus).toarray()
#dependent variable
y = dataset.iloc[:, 1].values

#********CLASSIFICATION**********

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


#Model 1: NAIVE BAYES*****

# Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_predNaive = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cmNaive = confusion_matrix(y_test, y_predNaive)

#Model2: Decision tree******

# Fitting classifier to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_predTree = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cmTree = confusion_matrix(y_test, y_predTree)

#Model3: Random forest*****

# Fitting Random Forest classification classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10, criterion = 'entropy',random_state=0 )
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_predForest = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cmForest = confusion_matrix(y_test, y_predForest)
