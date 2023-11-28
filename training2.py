import joblib
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained models
rf_model = joblib.load('random_forest_model.pkl')
nn_model = joblib.load('neural_network_model.pkl')
knn_model = joblib.load('knn_model.pkl')
svm_model = joblib.load('svm_model.pkl')

# Load the TF-IDF vectorizer
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Load the test data (replace 'your_test_data.csv' with the actual file name)
test_data = pd.read_csv('Code_Comment_Seed_Data.csv')  

# Assuming your test data has 'Comments' and 'Surrounding Code Context' columns
X_test_tfidf = vectorizer.transform(test_data['Comments'] + ' ' + test_data['Surrounding Code Context'])
y_test = test_data['Class']  # Replace 'Class' with the actual target column name

# Create an ensemble model
ensemble_model = VotingClassifier(estimators=[
    ('rf', rf_model),
    ('nn', nn_model),
    ('knn', knn_model),
    ('svm', svm_model)
], voting='hard')

# Fit the ensemble model on the test data
ensemble_model.fit(X_test_tfidf, y_test)

# Save the ensemble model
joblib.dump(ensemble_model, 'ensemble_model.pkl')

# Example usage of the saved ensemble model
new_user_comment = input("Enter the new comment: ")
new_user_code = input("Enter the new code: ")

# Vectorize the input text
new_input_vectorized = vectorizer.transform([new_user_comment + ' ' + new_user_code])

# Make predictions using the ensemble model
ensemble_prediction = ensemble_model.predict(new_input_vectorized)

print(f"Ensemble Model predicts: {'Useful' if ensemble_prediction[0] == 1 else 'Not Useful'}")
