import joblib

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

# Example usage
user_comment = input("Enter the comment: ")
user_code = input("Enter the code: ")

# Predict using the loaded Random Forest model
result_rf = predict_class(user_comment, user_code, loaded_rf_model, loaded_vectorizer)

# Print the result
print(f"The model predicts that the code-comment pair is in class: {result_rf}")
