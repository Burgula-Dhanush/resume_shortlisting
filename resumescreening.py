import os
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import spacy
from concurrent.futures import ThreadPoolExecutor

import time
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import JSONResponse
import json
app = FastAPI()
resume_folder= "H:/Resume/resumes"
@app.get("/resume_path")    
async def resume_path(path: str):
    # Specify the folder containing resumes
    resume_folder = path
    return  {'message':f"the path is {resume_folder}"}

def display(x, top_resume_indices, resume_files, cosine_similarities):
    result_data = {"Top_Resumes": []}

    for index in top_resume_indices:
        resume_file = resume_files[index]
        percentage_match = cosine_similarities[index] * 100
        result_data["Top_Resumes"].append({"Resume": resume_file, "Percentage_Match": percentage_match})

    return result_data

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF resume."""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page].extract_text()
    return text

def get_resume_texts(resume_folder):
    """Get a list of extracted texts from PDF resumes."""
    resume_files = [file for file in os.listdir(resume_folder) if file.endswith(".pdf")]
    
    with ThreadPoolExecutor() as executor:
        resume_texts = list(executor.map(lambda file: extract_text_from_pdf(os.path.join(resume_folder, file)), resume_files))
    
    return resume_texts, resume_files

def calculate_cosine_similarities(vectorizer, resume_texts, keywords):
    """Calculate cosine similarities between resumes and keywords."""
    tfidf_matrix = vectorizer.transform(resume_texts)
    cosine_similarities = linear_kernel(tfidf_matrix, vectorizer.transform([" ".join(keywords)]))
    return cosine_similarities.flatten()

def shortlist_and_display(vectorizer, cosine_similarities, x, resume_files):
    """Shortlist top x resumes based on similarity scores and display percentage match."""
    top_resume_indices = np.argsort(cosine_similarities)[-x:][::-1]
    
    result_data = display(x, top_resume_indices, resume_files, cosine_similarities)
    return {"message": result_data}

def extract_skills_nlp_based(job_description):
    """Extract skills from a job description using spaCy."""
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(job_description)
    
    # Extract nouns and proper nouns as potential skills
    extracted_skills = [token.text for token in doc if (token.pos_ == 'NOUN' or token.pos_ == 'PROPN')]
    return extracted_skills

@app.get("/enter_jd")    
async def main(job_description: str, x):
    # Get a list of extracted texts and resume files
    resume_texts, resume_files = get_resume_texts(resume_folder)

    # Extract skills from the job description
    job_description_skills = extract_skills_nlp_based(job_description)

    # Create a TfidfVectorizer and fit it only once
    vectorizer = TfidfVectorizer()
    vectorizer.fit_transform(resume_texts)
    if x=='all' or x=='All' or x=='ALL':
        x=len(resume_files)
    else:
        x=int(x)
    # Calculate cosine similarities and shortlist top resumes
    cosine_similarities = calculate_cosine_similarities(vectorizer, resume_texts, job_description_skills)
    result_json = shortlist_and_display(vectorizer, cosine_similarities, x, resume_files)
    
    return result_json

@app.get("/")    
async def home():
    return {'message':"localhost/enter_jd/?job_description=...&x=..  is the command to enter jd"}