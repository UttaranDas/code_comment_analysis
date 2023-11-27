# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 17:20:04 2023

@author: 97155
"""

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the ensemble model
ensemble_model = joblib.load('ensemble_model.pkl')

# Load the vectorizer (used for transforming input text into features)
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def preprocess_input(comment, code):
    # Add any additional preprocessing steps here if needed
    return comment + ' ' + code

def predict_usefulness(comment, code):
    # Preprocess the input
    input_text = preprocess_input(comment, code)

    # Vectorize the input text
    input_vectorized = vectorizer.transform([input_text])

    # Make predictions using the ensemble model
    prediction = ensemble_model.predict(input_vectorized)

    return "Useful" if prediction[0] == 1 else "Not Useful"

if __name__ == "__main__":
    # Example usage
    user_comment = input("Enter the comment: ")
    user_code = input("Enter the code: ")

    # Get predictions using the ensemble model
    prediction = predict_usefulness(user_comment, user_code)

    # Display the result
    print(f"The ensemble model predicts: {prediction}")
