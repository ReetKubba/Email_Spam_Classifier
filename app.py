import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk 
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


def transform_text(text):
    text=text.lower()#convert into lower
    text=nltk.word_tokenize(text)#convert into tokens       ['convert','into','token']
    
    y=[]
    for i in text:#removing special charecters like charecters other than alphabets and numbers
        if i.isalnum:
            y.append(i)
    text=y[:]
    y.clear()
    
    for i in text:#remove stopwords ie the words like i am that dont change the meaning of the sentence
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:#stemming that is converting words with same meaning to a single word like loving loved to love
        y.append(ps.stem(i))
    return " ".join(y)

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))
st.title('Email/SMS Spam Classifier')

input_sms=st.text_area("Enter the message")
if st.button('Predict'):
    #preprocess
    transformed_sms=transform_text(input_sms)
    #vectorize
    vector_input=tfidf.transform([transformed_sms])
    #predict
    result=model.predict(vector_input)[0]
    if result==1:
        st.header("Spam")
    else:
        st.header("Not Spam")
