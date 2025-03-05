from flask import Flask, render_template, request
import joblib
import pdfplumber
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
import os

app = Flask(__name__)

# Load trained model and vectorizer
model_path = "resume_classifier.pkl"
vectorizer_path = "tfidf_vectorizer.pkl"

if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
else:
    print("⚠️ ERROR: Model files not found. Check deployment.")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text if text else "⚠️ ERROR: No text extracted from PDF."
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

# Function to extract text from DOCX
def extract_text_from_docx(docx_file):
    try:
        doc = docx.Document(docx_file)
        return "\n".join([para.text for para in doc.paragraphs]) or "⚠️ ERROR: No text extracted from DOCX."
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return ""

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        uploaded_file = request.files["resume"]
        
        if not uploaded_file:
            print("⚠️ ERROR: No file uploaded.")
            return render_template("index.html", result="⚠️ No file uploaded.")

        print(f"✅ File uploaded: {uploaded_file.filename}")

        if uploaded_file.filename.endswith(".pdf"):
            resume_text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.filename.endswith(".docx"):
            resume_text = extract_text_from_docx(uploaded_file)
        else:
            print("⚠️ ERROR: Unsupported file format.")
            return render_template("index.html", result="⚠️ Unsupported file format.")

        if "ERROR" in resume_text:
            return render_template("index.html", result=resume_text)

        # Transform the resume text using the vectorizer
        transformed_text = vectorizer.transform([resume_text])
        prediction = model.predict(transformed_text)[0]

        result = "Shortlisted ✅" if prediction == 1 else "Not Shortlisted ❌"
        print(f"🔍 Prediction: {result}")
        return render_template("index.html", result=result)

    return render_template("index.html", result="")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
