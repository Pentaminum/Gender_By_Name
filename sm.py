import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

"""
Part 1: Preliminaries
"""

# Load the dataset
df = pd.read_csv("spam_messages.csv")

# Features and Labels
X = df["Contents"] # Features(names)
y = df["Labels"] # Labels(genders)

# Split the dataset into train and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)   
# print("Accuracy:", accuracy)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
# print("Confusion Matrix:\n", conf_matrix)

# Classification report
# print("Classification Report:\n", classification_report(y_test, y_pred))

# Map labels to binary values
y_prob = pipeline.predict_proba(X_test)[:, 1]
y_test_binary = y_test.map({'ham': 0, 'spam': 1})

# Plot precision-recall curve
precision, recall, _ = precision_recall_curve(y_test_binary, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.savefig('precision_recall_curve.png') 