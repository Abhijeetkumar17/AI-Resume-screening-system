import spacy

# Load spaCy NLP Model
nlp = spacy.load("en_core_web_sm")

resume_text = """
Abhijeet Kumar, Data Science Intern at CodeAlpha. 
Skills: Python, Java, C++, JavaScript, SQL, R, HTML, CSS, Tableau. 
Projects: Uber Data Analysis (Python, ML), Chat Bot (NLP, TensorFlow). 
Internships: Full Stack Web Developer (IBM, React, Java). 
Education: Lovely Professional University, B.Tech CSE (2022-2026), CGPA: 8.00.
"""

def extract_keywords(text):
    doc = nlp(text)
    keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN", "VERB"] and not token.is_stop]
    return list(set(keywords))

keywords = extract_keywords(resume_text)
print("Extracted Keywords:", keywords)

import pdfplumber
import docx

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()

def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text.strip()

