import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    # lower case
    text = text.lower()
    # tokenization
    words = nltk.word_tokenize(text)

    # removing special characters
    alpha_numeric = []
    for word in words:
        if word.isalnum():
            alpha_numeric.append(word)

    # removing stopwords and punctuation
    cleaned_words = []
    for word in alpha_numeric:
        if word not in stopwords.words('english') and word not in string.punctuation:
            cleaned_words.append(word)

    # stemming
    ps = PorterStemmer()
    for idx, word in enumerate(cleaned_words):
        cleaned_words[idx] = ps.stem(word)

    transformed_text = " ".join(cleaned_words)
    return transformed_text

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")