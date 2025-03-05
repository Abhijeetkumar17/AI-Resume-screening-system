import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Sample Dataset
data = {
    "resume": [
        "Data Scientist skilled in Python, SQL, Machine Learning.",
        "Full Stack Web Developer with React, Node.js, Java.",
        "Marketing expert with SEO, Google Ads, Digital Marketing.",
        "Software Engineer with Java, Spring Boot, Microservices.",
        "Data Science Intern experienced in Python, TensorFlow, data analytics."
    ],
    "hired": [1, 1, 0, 1, 1]  # 1 = Hired, 0 = Not Hired
}

df = pd.DataFrame(data)

# Convert text into numerical format using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["resume"])
y = df["hired"]

# Train Model
model = RandomForestClassifier()
model.fit(X, y)

# Save Model and Vectorizer
joblib.dump(model, "resume_classifier.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("✅ Model trained and saved successfully!")
