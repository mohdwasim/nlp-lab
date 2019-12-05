#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 10:41:13 2019

@author: dev
"""

import pickle
import numpy as np
import re
import nltk
from sklearn.datasets import load_files
from nltk.corpus import stopwords

nltk.download('stopwords')

# importing datasets

reviews= load_files('.')
X,y=reviews.data, reviews.target

with open('X.pickle','wb') as f:
    pickle.dump(X,f)

with open('y.pickle','wb') as f:
    pickle.dump(y,f)
    
with open('X.pickle', 'rb') as f:
    X=pickle.load(f)

with open('y.pickle', 'rb') as f:
    y=pickle.load(f)
corpus=[]

for i in range(0, len(X)):
    review=re.sub(r'\W',' ',str(X[i]))
    review=review.lower()
    review=re.sub(r'\s+[a-z]\s+',' ',str(X[i]))
    review=re.sub(r'^[a-z]\s+',' ',str(X[i]))
    review=re.sub(r'\s+',' ',str(X[i]))
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer(max_features=2000,min_df=3,max_df=.6,stop_words=stopwords.words('english'))
X=vectorizer.fit_transform(corpus).toarray()

from sklearn.feature_extraction.text import TfidfTransformer
transformer=TfidfTransformer()
X=transformer.fit_transform(X).toarray()


from sklearn.model_selection import train_test_split
text_train,text_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression 

classifier=LogisticRegression()
classifier.fit(text_train,y_train)

sen_prediction=classifier.predict(text_test)


from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,sen_prediction)


# saving classifier model

with open('classifier.pickle','wb') as f:
    pickle.dump(classifier,f)
    
# saving vectorizer model
    
with open('tfidfmodel.pickle','wb') as f:
    pickle.dump(vectorizer,f)

#Loading Saved classifier and vectorizer model
    

with open('classifier.pickle','rb') as f:
    clf= pickle.load(f)
    
    
with open('tfidfmodel.pickle','rb') as f:
    tfidf=pickle.load(f)
    
sample=['google is good']
sample = tfidf.transform(sample).toarray()

print(clf.predict(sample))