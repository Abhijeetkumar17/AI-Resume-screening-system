import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
import joblib

# Sample Training Data (Balanced with both 'Shortlisted' and 'Not Shortlisted')
data = {
    "resume": [
        "Data Scientist skilled in Python, SQL, Machine Learning.",
        "Full Stack Developer with React, Node.js, Java.",
        "Marketing expert with SEO, Google Ads, Digital Marketing.",
        "Software Engineer with Java, Spring Boot, Microservices.",
        "Python Developer with Flask, Django, AWS experience.",
        "Customer service representative with strong communication skills.",
        "Graphic designer with experience in Photoshop, Illustrator.",
        "Financial analyst skilled in accounting, taxation, Excel."
    ],
    "hired": [1, 1, 0, 1, 1, 0, 0, 0]  # 1 = Shortlisted, 0 = Not Shortlisted
}

df = pd.DataFrame(data)

# Balance dataset (Ensure equal number of shortlisted and not shortlisted)
df_hired = df[df["hired"] == 1]
df_not_hired = df[df["hired"] == 0]

df_not_hired_upsampled = resample(df_not_hired, replace=True, n_samples=len(df_hired), random_state=42)

df_balanced = pd.concat([df_hired, df_not_hired_upsampled])

# TF-IDF Vectorization (Fixes irrelevant word matching)
vectorizer = TfidfVectorizer(stop_words='english', min_df=2)
X = vectorizer.fit_transform(df_balanced["resume"])
y = df_balanced["hired"]

# Train Model
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)

# Save Model and Vectorizer
joblib.dump(model, "resume_classifier.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("✅ Model trained and saved successfully!")
