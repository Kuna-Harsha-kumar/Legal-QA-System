

# !pip install pandas numpy scikit-learn xgboost sentence-transformers transformers accelerate datasets openpyxl pdfplumber python-docx

from google.colab import files
uploaded = files.upload()

filename = list(uploaded.keys())[0]
print(f"✅ Uploaded file: {filename}")

import pandas as pd
import re

def load_any_document(file_path):
    text_content = ""

    if file_path.endswith((".xlsx", ".xls", ".csv")):
        df = pd.read_excel(file_path) if file_path.endswith((".xlsx", ".xls")) else pd.read_csv(file_path)
        df = df.fillna('')
        print(f"✅ Excel/CSV file loaded with shape {df.shape}. Columns: {list(df.columns)}")
        return "excel", df

    elif file_path.endswith(".pdf"):
        print("📘 Reading PDF...")
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("❌ The 'pdfplumber' library is required for reading PDF files. Please install it using '!pip install pdfplumber'.")
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text_content += page.extract_text() + "\n"
        print(f"✅ PDF loaded, {len(text_content)} characters extracted.")
        return "text", text_content

    elif file_path.endswith(".docx"):
        print("📘 Reading Word document...")
        try:
            from docx import Document
        except ImportError:
            raise ImportError("❌ The 'python-docx' library is required for reading Word files. Please install it using '!pip install python-docx'.")
        doc = Document(file_path)
        for para in doc.paragraphs:
            text_content += para.text + "\n"
        print(f"✅ Word file loaded, {len(text_content)} characters extracted.")
        return "text", text_content

    else:
        raise ValueError("❌ Unsupported file type. Please upload Excel, PDF, or Word.")

# Check if the uploaded file is supported before attempting to load
if filename.endswith((".xlsx", ".xls", ".csv", ".pdf", ".docx")):
    file_type, data = load_any_document(filename)
else:
    print(f"❌ Unsupported file type for {filename}. Please upload Excel, PDF, or Word.")

def preprocess_text(text):
    # Remove file references, page numbers, and junk symbols
    text = re.sub(r'Page\s*\d+[-–]?\d*', '', text)
    text = re.sub(r'EX[-\s]*\d+(\.\d+)?', '', text)
    text = re.sub(r'\b\d{4,}\b', '', text)  # remove long numeric tokens
    text = re.sub(r'[^a-zA-Z0-9\s.,;:()\-]', ' ', str(text))
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, chunk_size=100):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def extract_clean_clauses(file_type, data):
    clauses = []
    if file_type == "excel":
        for _, row in data.iterrows():
            text = ' '.join(map(str, row.values))
            cleaned = preprocess_text(text)
            if len(cleaned.split()) > 10:
                clauses.append(cleaned)
    else:
        text = preprocess_text(data)
        chunks = chunk_text(text, 100)
        clauses = [c for c in chunks if len(c.split()) > 10]
    print(f"✅ Extracted {len(clauses)} clean clauses/segments.")
    return clauses

clauses = extract_clean_clauses(file_type, data)

from sklearn.feature_extraction.text import TfidfVectorizer

print("🔹 Creating TF-IDF representations for clauses...")

# TF-IDF converts each clause into numeric vector form
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = vectorizer.fit_transform(clauses)

print(f"✅ TF-IDF matrix created with shape: {X_tfidf.shape}")

import xgboost as xgb
import numpy as np

print("🚀 Training XGBoost model on TF-IDF clause features...")

# 1️⃣ Prepare training data
y_dummy = np.mean(X_tfidf.toarray(), axis=1)  # simple numeric target
dtrain = xgb.DMatrix(X_tfidf, label=y_dummy)

# 2️⃣ Define parameters
params = {
    "objective": "reg:squarederror",
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "rmse",
    "verbosity": 0
}

# 3️⃣ Train
xgb_model = xgb.train(params, dtrain, num_boost_round=200)

print("✅ XGBoost model trained successfully!")

import matplotlib.pyplot as plt

xgb.plot_importance(xgb_model, max_num_features=15)
plt.title("Top 15 Important TF-IDF Features Learned by XGBoost")
plt.show()

import xgboost as xgb

print("🚀 Preparing XGBoost-compatible matrix (no supervised training)...")

# Convert TF-IDF to DMatrix for fast internal representation
dtrain = xgb.DMatrix(X_tfidf)

print(f"✅ DMatrix built successfully with {dtrain.num_row()} rows and {dtrain.num_col()} columns.")

from sklearn.metrics.pairwise import cosine_similarity
import re

def clean_query(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return text.lower().strip()

def ask_question(question, vectorizer, X_tfidf, clauses, top_k=3):
    q_clean = clean_query(question)
    q_vec = vectorizer.transform([q_clean])
    sims = cosine_similarity(q_vec, X_tfidf).flatten()
    top_idx = sims.argsort()[-top_k:][::-1]

    print(f"\n💬 Question: {question}")
    print("📄 Top related clauses:\n")
    for i, idx in enumerate(top_idx):
        print(f"🔹 Clause {i+1} (Score={sims[idx]:.3f}):\n{clauses[idx]}\n{'-'*80}")

questions = [
    "What is the minimum commitment?",
    "Are there any penalties for breach or termination?",
    "What is the warranty period?",
    "What does the agreement cover?"
]

for q in questions:
    ask_question(q, vectorizer, X_tfidf, clauses)

from sentence_transformers import SentenceTransformer

print("🧠 Building legal-specific embeddings...")
# Alternative legal models: "nlpaueb/legal-bert-base-uncased" or "sentence-transformers/all-mpnet-base-v2"
embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
clause_embeddings = embedder.encode(clauses, show_progress_bar=True)
print(f"✅ Created embeddings for {len(clause_embeddings)} clauses.")

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

print("📦 Loading FLAN-T5 model...")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
llm = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
print("✅ Model loaded successfully.")

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

def retrieve_related_clauses(question, embedder, clauses, clause_embeddings, llm=None, tokenizer=None, top_k=3, summarize=True, min_score=0.001):
    q_vec = embedder.encode([question])
    sims = cosine_similarity(q_vec, clause_embeddings).flatten()

    # Filter out weak matches
    strong_idx = np.where(sims >= min_score)[0]
    if len(strong_idx) == 0:
        print(f"\n💬 Question: {question}\n❌ No clauses found with similarity above {min_score}. Try rephrasing.")
        return

    top_idx = strong_idx[np.argsort(sims[strong_idx])[-top_k:][::-1]]

    print(f"\n💬 Question: {question}")
    print(f"📑 Retrieved {len(top_idx)} Related Clauses (Score ≥ {min_score}):\n")

    combined_text = ""
    for i, idx in enumerate(top_idx):
        print(f"🔹 Clause {i+1} (Score={sims[idx]:.3f}):\n")
        print(f"\"{clauses[idx]}\"\n")
        print("-" * 80)
        combined_text += clauses[idx] + " "

    if summarize and llm and tokenizer:
        prompt = f"""
        You are a legal assistant. Based on the following extracted clauses, answer the question precisely
        and summarize the relevant legal meaning professionally.

        Clauses:
        {combined_text}

        Question: {question}
        Legal Summary:
        """

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = llm.generate(**inputs, max_length=256, temperature=0.6, top_p=0.9)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\n🤖 Legal Interpretation:\n")
        print(answer)

# Try these first
questions = [
    "What is the minimum commitment?",
    "Are there any penalties for breach or termination?",
    "What is the warranty period?",
    "What does the agreement cover?"
]

for q in questions:
    retrieve_related_clauses(q, embedder, clauses, clause_embeddings, llm, tokenizer)