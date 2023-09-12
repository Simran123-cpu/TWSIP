#!/usr/bin/env python
# coding: utf-8

# # Detecting Fake new with python and Machine learning

# # About Detecting Fake News with python:
# This advanced python project of detecting fake news deals with fake and real news . using sklearn, we build tfidVectorizer on our dataset.then, we initialize a PassiveAggresive Classifier fit the model.

# necessary libraries to be imoported

# In[5]:


get_ipython().system('pip install sklearn')


# In[10]:


import numpy as np
import pandas as pd
import itertools 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix 


# In[11]:


df=pd.read_csv(r"C:\Users\DELL\Documents\news.csv")
df.shape
df.head()


# In[13]:


labels=df.label
labels.head()


# In[14]:


#DataFlair - Split the dataset
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)


# In[15]:


#DataFlair - Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

#DataFlair - Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)


# In[16]:


#DataFlair - Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

#DataFlair - Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# In[17]:


#DataFlair - Build confusion matrix
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])


# so with this model we have 589 true positives,585 true negatives , 44 false positives and 49 false negatives.

# In[ ]:




