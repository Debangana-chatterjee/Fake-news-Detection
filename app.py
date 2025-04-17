import streamlit as st
import joblib
import pandas as pd
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
stopwords=nltk.corpus.stopwords.words('english')
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer=WordNetLemmatizer()
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def cleanData(row):
    row = row.lower()
    row = re.sub('[^a-zA-Z]',' ',row)
    token = row.split()
    removeStop = [i for i in token if i not in stopwords]
    lemm_text_temp = [wordnet_lemmatizer.lemmatize(word,pos='v') for word in removeStop]
    lemm_text = [wordnet_lemmatizer.lemmatize(word,pos='n') for word in lemm_text_temp]
    cleaned_string = ""
    for word in lemm_text:
        cleaned_string+=word
        cleaned_string+= ' '
    return cleaned_string

model = joblib.load('MNB')
st.title("Fake News")
ip=st.text_input("Enter message: ")
op=model.predict([cleanData(ip)])

if st.button("Detect"):
    if op==1:
        final='Real'
    else:
        final='Fake'
    st.subheader(f"The above news snippet is **{final}**.")

