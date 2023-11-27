import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load the data
data = pd.read_csv('FIRE2023_IRSE_training_Code_Comment_Seed_Data.csv')

# Data Preprocessing
def clean_text(text):
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    text = text.lower()
    return text

def preprocess_input(comment, code):
    comment = clean_text(comment)
    code = clean_text(code)
    return comment + ' ' + code

# Handle missing values
data.dropna(inplace=True)

# Text cleaning
data['Comments'] = data['Comments'].apply(clean_text)
data['Surrounding Code Context'] = data['Surrounding Code Context'].apply(clean_text)

# Outlier removal
z_scores_comments = (data['Comments'].apply(len) - data['Comments'].apply(len).mean()) / data['Comments'].apply(len).std()
z_scores_code = (data['Surrounding Code Context'].apply(len) - data['Surrounding Code Context'].apply(len).mean()) / data['Surrounding Code Context'].apply(len).std()

data = data[(z_scores_comments.abs() < 3) & (z_scores_code.abs() < 3)]

# Remove preceding numbers from 'Surrounding Code Context'
data['Surrounding Code Context'] = data['Surrounding Code Context'].str.replace(r'\b\d+\b', '')

# Vectorization
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(data['Comments'] + ' ' + data['Surrounding Code Context'])
y = data['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Model Training - Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Save the trained Random Forest model and vectorizer
joblib.dump(rf_classifier, 'new/trained_model_rf.pkl')
joblib.dump(vectorizer, 'new/tfidf_vectorizer_rf.pkl')

# Model Evaluation
rf_predictions = rf_classifier.predict(X_test)
rf_report = classification_report(y_test, rf_predictions)
print("Random Forest Classification Report:\n", rf_report)
