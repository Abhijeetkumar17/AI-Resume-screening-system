from flask import Flask, request, render_template
import joblib
import pdfplumber
import docx

app = Flask(__name__)

# Load Trained Model and Vectorizer
model = joblib.load("resume_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()

# Function to extract text from DOCX
def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text.strip()

# ✅ FIXED: Ensure "/" Route Sends `result`
@app.route("/", methods=["GET", "POST"])
def home():
    result = ""  # Default value
    if request.method == "POST":
        if "resume_text" in request.form:
            resume_text = request.form["resume_text"]
            transformed_text = vectorizer.transform([resume_text])
            prediction = model.predict(transformed_text)[0]
            result = "Shortlisted ✅" if prediction == 1 else "Not Shortlisted ❌"

    return render_template("index.html", result=result)

# ✅ FIXED: Ensure "/upload" Route Sends `result`
@app.route("/upload", methods=["POST"])
def upload():
    result = ""  # Default value
    if "resume" not in request.files:
        return render_template("index.html", result="No file uploaded")

    uploaded_file = request.files["resume"]
    if uploaded_file.filename.endswith(".pdf"):
        resume_text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.filename.endswith(".docx"):
        resume_text = extract_text_from_docx(uploaded_file)
    else:
        return render_template("index.html", result="Unsupported file format")

    transformed_text = vectorizer.transform([resume_text])
    prediction = model.predict(transformed_text)[0]
    result = "Shortlisted ✅" if prediction == 1 else "Not Shortlisted ❌"
    
    return render_template("index.html", result=result)

# Run Flask App
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=10000)
