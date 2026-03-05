import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load fake dataset
data = pd.read_csv("sample_data.csv")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data['tweet_text'], data['label'], test_size=0.25, random_state=42
)

# Feature extraction
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model training
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train_vec, y_train)
# trained with 10 trees

# Evaluation
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))