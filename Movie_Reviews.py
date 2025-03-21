import streamlit as st
import pickle
import re
import numpy as np

# Load models and vectorizer
logistic_model = pickle.load(open('logistic_regression_model.sav', 'rb'))
nb_model = pickle.load(open('multinomial_nb_model.sav', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.sav', 'rb'))

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Function to predict sentiment
def predict_sentiment(review, model_choice):
    cleaned_review = preprocess_text(review)  # Preprocess input
    review_tfidf = tfidf_vectorizer.transform([cleaned_review])

    if model_choice == 'Logistic Regression':
        prediction = logistic_model.predict(review_tfidf)
    else:
        prediction = nb_model.predict(review_tfidf)

    return "Positive 😊" if prediction[0] == 1 else "Negative 😞"

# Streamlit UI
st.set_page_config(page_title="Movie Review Sentiment Analysis", page_icon="🎬", layout="centered")

# Custom Styling
st.markdown("""
    <style>
        .title {text-align: center; color: #f64c72; font-size: 36px; font-weight: bold;}
        .stTextArea textarea {background-color: #f8f9fa; font-size: 18px; border-radius: 10px; padding: 10px;}
        .stButton button {background-color: #f64c72; color: white; font-size: 18px; border-radius: 10px; padding: 10px 20px;}
        .stButton button:hover {background-color: #ff758f;}
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='title'>Movie Review Sentiment Analysis 🎥</h1>", unsafe_allow_html=True)

# User input
review = st.text_area("Enter your movie review:", height=150)
model_choice = st.selectbox("Choose a model:", ["Logistic Regression", "Multinomial Naive Bayes"])

if st.button("Predict Sentiment"):
    if review.strip():
        sentiment = predict_sentiment(review, model_choice)
        st.success(f"Predicted Sentiment: {sentiment}")

        # Debugging output to check predictions
        print(f"User Input: {review}")
        print(f"Processed Input: {preprocess_text(review)}")
        print(f"Model Used: {model_choice}")
        print(f"Prediction: {sentiment}")

    else:
        st.warning("Please enter a review before predicting.")

# Debugging: Check TF-IDF vectorizer feature names in terminal
print("Sample TF-IDF Features:", tfidf_vectorizer.get_feature_names_out()[:20])
