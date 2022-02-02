import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import os
import time

import docx2txt
from PyPDF2 import PdfFileReader
import pdfplumber
import textract
import shutil

ps = PorterStemmer()

tfidf = pickle.load(open('tfidf.pkl','rb'))
model = pickle.load(open('model_random_forest.pkl','rb'))

def prediction(text):
    pred=classifier.predict(text) 
    return pred
	
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

def main():
    
    st.title("Sentiment Analyzer")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:black;text-align:center;">Streamlit Sentiment Analyzer ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    review = st.text_input("Input text",placeholder="Type your review here")
    text = transform_text(review)
    #pred = prediction(text)
    vector_input = tfidf.transform([text])
    #3. predict
    #model = model.fit()
    pred = model.predict(vector_input)[0]
    result=''
    if st.button("Predict"):
        with st.spinner('Wait...'):
            time.sleep(2)
                
            if pred=='0':
                result='This review is bad! Don not buy this product!!'
            elif pred=='1':
                st.balloons()
                result= 'This review is good! Just buy this product!!'
                
        st.success(result)
	

if __name__=='__main__':
    main()
	