import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# UI
st.title("📄 Resume Screening System")

job_desc = st.text_area("Enter Job Description")

uploaded_files = st.file_uploader(
    "Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True
)

if st.button("Screen Resumes"):
    if not job_desc or not uploaded_files:
        st.warning("Please provide job description and upload resumes")
    else:
        resumes = []
        names = []

        for file in uploaded_files:
            text = extract_text(file)
            resumes.append(text)
            names.append(file.name)

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([job_desc] + resumes)

        # Similarity
        scores = cosine_similarity(vectors[0:1], vectors[1:])[0]

        # Display results
        results = list(zip(names, scores))
        results.sort(key=lambda x: x[1], reverse=True)

        st.subheader("📊 Ranking Results")

        for name, score in results:
            st.write(f"**{name}** → Score: {score:.2f}")