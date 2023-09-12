#!/usr/bin/env python
# coding: utf-8

# # Movie Recommendation System

# Recommender System is a system that seeks to predict or filter preferences according to the user’s choices. Recommender systems are utilized in a variety of areas including movies, music, news, books, research articles, search queries, social tags, and products in general. 

# Libraries to be imported :

# In[1]:


import numpy as np
import pandas as pd 


# In[2]:


column_names = ['user_id', 'item_id', 'rating', 'timestamp']


# In[3]:


df = pd.read_csv("C://Users//DELL//Downloads//dataset (2).csv", sep = '\t', names = column_names)


# In[4]:


df.head()


# In[5]:


movie_titles=pd.read_csv("C://Users//DELL//Downloads//movieIdTitles.csv")


# In[6]:


movie_titles.head()


# In[7]:


df=pd.merge(df,movie_titles)


# In[8]:


df.head()


# # EDA

# Import vizualisation libraries

# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


sns.set_style('white')


# In[11]:


df.groupby('title')['rating'].mean().sort_values(ascending=False).head(10)


# In[12]:


df.groupby('title')['rating'].count().sort_values(ascending=False).head(10)


# In[13]:


ratings=pd.DataFrame(df.groupby('title')['rating'].mean())


# In[14]:


ratings.head()


# In[15]:


ratings['rating_numbers'] = pd.DataFrame(df.groupby('title')['rating'].count())


# In[16]:


ratings.head()


# In[17]:


ratings['rating_numbers'].hist(bins=70)


# In[18]:


ratings['rating'].hist(bins=70)


# In[19]:


sns.jointplot(x='rating',y='rating_numbers',data=ratings,alpha=0.5)


# # Create the Recommendation System

# In[20]:


moviemat=df.pivot_table(index='user_id',columns='title',values='rating')


# In[21]:


moviemat.head()


# In[22]:


ratings.sort_values('rating_numbers', ascending = False).head(10)


# In[26]:


starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']
  
starwars_user_ratings.head()


# Finding correlation of a movie using corrwith()function
# 

# In[27]:


similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)
  
corr_starwars = pd.DataFrame(similar_to_starwars, columns =['Correlation'])
corr_starwars.dropna(inplace = True)
  
corr_starwars.head()


# In[29]:


corr_liarliar = pd.DataFrame(similar_to_liarliar, columns =['Correlation'])
corr_liarliar.dropna(inplace = True)
  
corr_liarliar = corr_liarliar.join(ratings['rating_numbers'])
corr_liarliar[corr_liarliar['rating_numbers']>100].sort_values('Correlation', ascending = False).head()

