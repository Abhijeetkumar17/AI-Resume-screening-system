import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import re

# Sample Training Data (More balanced)
data = {
    "resume": [
        "Python Developer with 3+ years experience in Django and Flask.",
        "Marketing Manager experienced in SEO, Google Ads, and PPC campaigns.",
        "Machine Learning Engineer with TensorFlow, PyTorch, and NLP experience.",
        "Graphic Designer skilled in Adobe Photoshop, Illustrator, and UI/UX design.",
        "Software Engineer with expertise in Java, Spring Boot, and Microservices.",
        "Entry-level student looking for a software engineering internship.",
        "Data Entry Clerk skilled in Microsoft Excel and administrative tasks.",
        "Customer Support Representative with experience in handling customer complaints."
    ],
    "hired": [1, 1, 1, 1, 1, 0, 0, 0]  # 1 = Shortlisted, 0 = Not Shortlisted
}

df = pd.DataFrame(data)

# Data Cleaning Function
def clean_text(text):
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)  # Remove special characters
    return text.lower().strip()

df["resume"] = df["resume"].apply(clean_text)

# Train TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words="english", min_df=2)
X = vectorizer.fit_transform(df["resume"])
y = df["hired"]

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save Model and Vectorizer
joblib.dump(model, "resume_classifier.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("✅ Model trained and saved successfully!")
