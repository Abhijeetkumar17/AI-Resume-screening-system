import spacy
import pdfplumber
import docx
import joblib
from flask import Flask, render_template, request
import os

app = Flask(__name__)

# Load ML Model & TF-IDF Vectorizer
model_path = "resume_classifier.pkl"
vectorizer_path = "tfidf_vectorizer.pkl"

if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
else:
    print("⚠️ ERROR: Model files not found!")

# Load spaCy NLP Model
nlp = spacy.load("en_core_web_sm")

# ✅ Function to Extract Keywords
def extract_keywords(text):
    doc = nlp(text)
    keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN", "VERB"] and not token.is_stop]
    return list(set(keywords))

# ✅ Function to Extract Text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text.strip()

# ✅ Function to Extract Text from DOCX
def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    return "\n".join([para.text for para in doc.paragraphs]).strip()

# ✅ Function to Match Keywords Against Required Skills
def match_keywords(resume_keywords, required_keywords):
    matched = set(resume_keywords) & set(required_keywords)
    score = len(matched) / len(required_keywords)  # Match percentage
    return "Shortlisted ✅" if score > 0.5 else "Not Shortlisted ❌"

# ✅ List of Required Keywords (Modify Based on Job Role)
required_keywords = ["Python", "Machine Learning", "SQL", "Tableau", "Django", "Flask"]

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        uploaded_file = request.files["resume"]

        if not uploaded_file:
            return render_template("index.html", result="⚠️ No file uploaded.")

        if uploaded_file.filename.endswith(".pdf"):
            resume_text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.filename.endswith(".docx"):
            resume_text = extract_text_from_docx(uploaded_file)
        else:
            return render_template("index.html", result="⚠️ Unsupported file format.")

        # Extract Keywords
        resume_keywords = extract_keywords(resume_text)

        # 🔍 ML-Based Shortlisting
        transformed_text = vectorizer.transform([resume_text])
        ml_prediction = model.predict(transformed_text)[0]
        ml_result = "Shortlisted ✅" if ml_prediction == 1 else "Not Shortlisted ❌"

        # 🔍 Keyword-Based Shortlisting
        keyword_result = match_keywords(resume_keywords, required_keywords)

        # ✅ Final Decision: If Either ML or Keyword Matching Shortlists the Candidate
        final_result = "Shortlisted ✅" if (ml_result == "Shortlisted ✅" or keyword_result == "Shortlisted ✅") else "Not Shortlisted ❌"

        return render_template("index.html", result=final_result)

    return render_template("index.html", result="")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
