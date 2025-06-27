import pickle
import streamlit as st
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os
nltk.download('popular')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower() #lowercase
    text = nltk.word_tokenize(text) #tokenize

    y = []
    for i in text:
        if i.isalnum():
            y.append(i) #removing special characters

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i) #removing stopwords and punctuations

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_input("Enter The Message")

if st.button("Predict"):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result==1:
        st.header("Spam")
    else:
        st.header("Not Spam")    

