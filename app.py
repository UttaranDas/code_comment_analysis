# app.py
import joblib
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# Load the trained Random Forest model and vectorizer
loaded_rf_model = joblib.load('trained_model_rf.pkl')
loaded_vectorizer = joblib.load('tfidf_vectorizer_rf.pkl')

# Data Preprocessing
def clean_text(text):
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    text = text.lower()
    return text

def preprocess_input(comment, code):
    comment = clean_text(comment)
    code = clean_text(code)
    return comment + ' ' + code

# Prediction function for user input
def predict_class(comment, code, model, vectorizer):
    input_text = preprocess_input(comment, code)
    input_vectorized = vectorizer.transform([input_text])

    prediction = model.predict(input_vectorized)
    return prediction[0]


def main():
    # Add your logo image
    logo_image = st.image("Vector.svg", use_column_width=False, width=100)

    st.title("Code Usefulness Predictor")

    # Use st.columns to create two columns
    col1, col2 = st.columns(2, gap="medium")

    # Text box for code input in the first column
    with col1:
        code_input = st.text_area("", "", height=391, placeholder="Paste code here")

    # Text box for comment input in the second column
    with col2:
        comment_input = st.text_area("", "", height=391, placeholder="Enter comment here")

    # Button to trigger the prediction
    if st.button("Predict Usefulness"):
        if code_input and comment_input:
            preprocess_input(comment_input, code_input)
            result_rf = predict_class(comment_input, code_input, loaded_rf_model, loaded_vectorizer)
            st.write(f"Prediction: {result_rf}")
        else:
            st.warning("Please enter code and comment.")

if __name__ == "__main__":
    main()
