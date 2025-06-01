from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import sklearn 
import joblib
from sklearn.feature_extraction.text import CountVectorizer
import spacy
import nltk
import re
from nltk.corpus import stopwords
stop=stopwords.words("english")
import string
from nltk.stem import PorterStemmer
pos_stem=PorterStemmer()

fitdata=joblib.load(r"fit_data")
cv=CountVectorizer(max_features=1500)
x=cv.fit(fitdata)

Model=joblib.load(r"review_prediction_model")

def clean(text):
       review=text.lower()
       review=re.sub('[^a-zA-z]',' ',review)
       review=review.split()    
       review=[i for i in review  if i not in stop]
       review=[ i for i in review if i not in string.punctuation]
       review=[pos_stem.stem(word) for word in review]
       review=" ".join(review)
       return review


app= Flask(__name__)

@app.route("/")
def index():
    return(render_template("index.html"))

@app.route("/predict",methods=["POST","GET"])
def predict():
    text=(request.form["input_given"])
    text=str(text)
    Text=clean(text)
    input_vec=cv.transform([Text])
    prediction=Model.predict(input_vec)
    Prediction=prediction[0]
    Prediction=str(Prediction)
    
    if Prediction=="Liked":
        return render_template("index.html",Prediction="👍")
       
    elif Prediction=="Disliked":
        return render_template("index.html",Prediction="👎")




if __name__=="__main__":
    app.run(debug=True)

