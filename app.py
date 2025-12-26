from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

internships = [
    # 🧠 Artificial Intelligence & Machine Learning
    {"role": "Machine Learning Intern", "company": "Google AI", "location": "Bengaluru, India",
     "skills": ["AI", "ML", "Deep Learning", "NLP", "Computer Vision"],
     "interests": ["Artificial Intelligence", "Neural Networks", "Deep Learning Applications"],
     "link": "https://ai.google/research/join-us"},
    {"role": "AI Research Intern", "company": "Microsoft Research", "location": "Hyderabad, India",
     "skills": ["AI", "ML", "Computer Vision", "Reinforcement Learning"],
     "interests": ["AI Research", "Robotics", "Computer Vision Projects"],
     "link": "https://www.microsoft.com/en-us/research/careers/"},
    {"role": "Deep Learning Intern", "company": "OpenAI", "location": "Remote",
     "skills": ["Deep Learning", "NLP", "Transformers"],
     "interests": ["Natural Language Processing", "AI Research", "Transformers"],
     "link": "https://openai.com/careers"},
    # ... (more internship entries) ...
    # 🧬 Bioinformatics & Computational Biology internships
    {"role": "Bioinformatics Intern", "company": "NCBI", "location": "Bethesda, USA",
     "skills": ["Genomics", "Python", "R", "Data Analysis"],
     "interests": ["Computational Biology", "Genomics", "Healthcare AI"],
     "link": "https://www.ncbi.nlm.nih.gov/about/careers/"}
]

model = None
chat_session = None


def preprocess(text):
    if not text:
        return ""
    df = pd.DataFrame({"input": [text]})
    df["input"] = (df["input"]
                   .str.lower()
                   .str.replace(r"[^a-z0-9\s]", " ", regex=True)
                   .str.replace(r"\s+", " ", regex=True)
                   .str.strip())
    return df["input"].iloc[0]


def recommend_internships(user_input):
    user_input = preprocess(user_input)
    corpus = [" ".join(intern["skills"] + intern["interests"]) for intern in internships]
    corpus.append(user_input)
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarity = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    
    results = []
    for idx, score in enumerate(similarity[0]):
        results.append({**internships[idx], "match": round(score * 100, 2)})
        
    return sorted(results, key=lambda x: x["match"], reverse=True)


@app.route("/")
def serve_frontend():
    return send_from_directory(".", "index.html")


@app.route("/recommend", methods=["POST"])
def recommend():
    user_input = request.json.get("skills", "")
    recommendations = recommend_internships(user_input)
    return jsonify(recommendations)


@app.route("/chat", methods=["POST"])
def chat():
    global chat_session
    user_message = request.json.get("message", "")
    
    if chat_session is None:
        return jsonify({"reply": "AI is not initialized. Please check the server logs."})
    
    try:
        response = chat_session.send_message(user_message)
        return jsonify({"reply": response.text})
    except Exception:
        return jsonify({"reply": "Sorry, something went wrong with the AI response."})


if __name__ == "__main__":
    try:
        api_key = "AIzaSyBWUBuNOdXL0nOu8u927FhX0wEbq8rEi58"
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        chat_session = model.start_chat(history=[])
        print("AI model successfully initialized.")
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        exit()

    app.run(host="0.0.0.0", port=8674)
