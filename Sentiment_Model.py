import numpy as np 
import pandas as pd 
import nltk
import spacy
import re 
import seaborn as sns
import string
nltk.download("stopwords")
from nltk.corpus import stopwords
stop=stopwords.words("english")
from nltk.stem import PorterStemmer
pos_stem=PorterStemmer()
import joblib
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import  CountVectorizer
import base64



fitdata=joblib.load(r"fit_data")
cv=CountVectorizer(max_features=1500)
x=cv.fit(fitdata)




st.title(':rainbow[Sentiment Prediction On Food Reviews]')
st.markdown(
    """
    <style>
    .stTextInput{
        position: relative;
        top: 230px;  
        left: 10px;
    }
    .stButton {
        position: relative;
        top: 173px;  
        left: 720px;
     
       
    }
    </style>
    """, 
    unsafe_allow_html=True
)
input=st.text_input(":red[enter your comment >>]")
button=st.button(":violet[Predict]")
image_path = r"Diving_Into_the_Customer_Satisfaction_Survey_hero.png"
with open(image_path,"rb") as file:
   image_data=base64.b64encode(file.read()).decode()
page_element=f"""
<style>
[data-testid="stAppViewContainer"]
{{
  background-image:url("data:image;base64,{image_data}");
  background-size:10x 10px;
  background-position:center;
  background-repeat:no-repeat;

 
}}
<style>
"""
st.markdown(page_element,unsafe_allow_html=True)
if button:
    def clean(text):
       review=text.lower()
       review=re.sub('[^a-zA-z]',' ',review)
       review=review.split()    
       review=[ i for i in review if i not in string.punctuation]
       review=[pos_stem.stem(word) for word in review]
       review=" ".join(review)
    
       return review
    input_vec=cv.transform([input])
    model=joblib.load(r"review_prediction_model")
    prediction=model.predict(input_vec)
    if prediction=="Disliked":
       st.subheader("👎Disliked")
    if prediction=="Liked":
       st.subheader("👍Liked")
    

