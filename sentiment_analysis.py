import numpy as np 
import pandas as pd 
import spacy
import nltk
import re 
import seaborn as sns
import string
from nltk.corpus import stopwords
stop=stopwords.words("english")
from nltk.stem import PorterStemmer
pos_stem=PorterStemmer()
import joblib
import streamlit as st





df=pd.read_csv(r"H:\reviews\Restaurant_Reviews.csv")




df


# In[4]:


df.drop_duplicates(inplace=True)


# In[5]:


df.duplicated().sum()


# In[6]:


df["Liked"]=df["Liked"].map({1:"Liked",0:"Disliked"})


# In[7]:


df


# In[8]:


df["Liked"].value_counts().plot(kind='bar')


# In[9]:


df["len"]=df["Review"].apply(len)


# In[10]:


df


# In[11]:


df.query('Liked=="Disliked"')["len"].mean()


# In[12]:

a=df.head(10).plot



# In[13]:


sns.barplot(x=df["Liked"],y=df["len"])


# In[14]:


def clean(text):
    review=text.lower()
    review=re.sub('[^a-zA-z]',' ',review)
    review=review.split()    
    review=[ i for i in review if i not in string.punctuation]
    review=[pos_stem.stem(word) for word in review]
    review=" ".join(review)
    
    return review
    


# In[15]:


df["cleaned_text"]=df["Review"].apply(lambda X:clean(X))


# In[16]:


df


# In[17]:


x=df["cleaned_text"]
y=df["Liked"]


# In[19]:


from sklearn.feature_extraction.text import CountVectorizer


# In[20]:


cv=CountVectorizer(max_features=1500)


# In[21]:


cv.fit(x)


# In[22]:


X=cv.transform(x)


# In[23]:


from sklearn.model_selection import train_test_split


# In[24]:


xtrain,xtest,ytrain,ytest=train_test_split(X,y,train_size=.80,random_state=0)


# In[25]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report


# In[26]:


LR=LogisticRegression()
DT=DecisionTreeClassifier()
RDF=RandomForestClassifier()
SVC=SVC()
GB=GaussianNB()


# In[27]:


LR.fit(xtrain,ytrain)


# In[28]:


print(accuracy_score(ytrain,LR.predict(xtrain)))
print(accuracy_score(ytest,LR.predict(xtest)))
print(classification_report(ytest,LR.predict(xtest)))


# In[29]:


DT.fit(xtrain,ytrain)


# In[30]:


print(accuracy_score(ytrain,DT.predict(xtrain)))
print(accuracy_score(ytest,DT.predict(xtest)))


# In[31]:


RDF.fit(xtrain,ytrain)


# In[32]:


print(accuracy_score(ytrain,RDF.predict(xtrain)))
print(accuracy_score(ytest,RDF.predict(xtest)))
print(classification_report(ytest,RDF.predict(xtest)))


