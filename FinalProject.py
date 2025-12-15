#!pip install -q pandas numpy scikit-learn xgboost sentence-transformers transformers accelerate datasets openpyxl pdfplumber python-docx rank_bm25 rouge_score peft
#!pip install bert-score
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from google.colab import files
from collections import Counter
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import Dataset
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from bert_score import score as bertscore
import torch

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')


uploaded = files.upload()

filename = list(uploaded.keys())[0]
print(f"Uploaded file: {filename}")

def load_any_document(file_path):
    file_path_lower = file_path.lower()
    text_content = ""
    # -------- Excel / CSV --------
    if file_path_lower.endswith((".xlsx", ".xls", ".csv")):
        try:
            if file_path_lower.endswith(".csv"):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            df = df.fillna('')
            print(f"Excel/CSV file loaded with shape {df.shape}. Columns: {list(df.columns)}")
            return "excel", df
        except Exception as e:
            raise ValueError(f"Failed to read spreadsheet file: {e}")
    # -------- PDF --------
    elif file_path_lower.endswith(".pdf"):
        print("Reading PDF...")
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content += page_text + "\n"
        print(f"PDF loaded, {len(text_content)} characters extracted.")
        return "text", text_content
    # -------- Word (.docx) --------
    elif file_path_lower.endswith(".docx"):
        print(" Reading Word document...")
        from docx import Document
        doc = Document(file_path)
        for para in doc.paragraphs:
            text_content += para.text + "\n"
        print(f"Word file loaded, {len(text_content)} characters extracted.")
        return "text", text_content
    # -------- Plain text (optional) --------
    elif file_path_lower.endswith(".txt"):
        print(" Reading plain text file...")
        with open(file_path, "r", errors="ignore") as f:
            text_content = f.read()
        print(f"TXT loaded, {len(text_content)} characters extracted.")
        return "text", text_content
    else:
        raise ValueError(f"Unsupported file type: {file_path}\nAllowed: Excel, Word, PDF, TXT")

if filename.lower().endswith((".xlsx", ".xls", ".csv", ".pdf", ".docx", ".txt")):
    file_type, data = load_any_document(filename)
else:
    raise ValueError(f"Unsupported file type: {filename}. Please upload Excel, PDF, Word, or TXT.")

def preprocess(text: str) -> str:
    text = re.sub(r'Page\s*\d+[-–]?\d*', '', text)
    text = re.sub(r'EX[-\s]*\d+(\.\d+)?', '', text)
    text = re.sub(r'\b\d{4,}\b', '', text)  # remove long numeric tokens (years, etc.)
    text = re.sub(r'[^a-zA-Z0-9\s.,;:()\-]', ' ', str(text))
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_clean_clauses(file_type, data):
    clauses = []
    if file_type == "excel":
        # join all columns in each row
        for _, row in data.iterrows():
            row_text = " ".join([str(v) for v in row.values if str(v).strip()])
            clean = preprocess(row_text)
            for s in sent_tokenize(clean):
                if len(s.split()) > 5:
                    clauses.append(s.strip())
    else:
        clean = preprocess(data)
        for s in sent_tokenize(clean):
            if len(s.split()) > 5:
                clauses.append(s.strip())

    print(f"Extracted {len(clauses)} cleaned clauses")
    return clauses

# get raw text (needed later for heading-based labels)
if file_type == "text":
    full_text = data
else:
    # rebuild a rough "text" from document
    if isinstance(data, pd.DataFrame):
        full_text = "\n".join([" ".join(map(str, row)) for _, row in data.iterrows()])
    else:
        full_text = str(data)
clauses = extract_clean_clauses(file_type, data)
if len(clauses) == 0:
    raise ValueError(" No clauses extracted. Check the input format or preprocessing.")

def extract_headings_general(text: str):
    lines = text.split("\n")
    labels = []
    for line in lines:
        raw = line.strip()
        if not raw:
            continue
        # ARTICLE / SECTION headings
        if re.match(r"^(ARTICLE|SECTION)\s+[0-9A-Za-z.\-]+", raw, re.IGNORECASE):
            labels.append(raw)
            continue
        # Numbered headings (1., 1.1., 2.3.4)
        if re.match(r"^[0-9]+(\.[0-9]+)*[.)]?\s+.+", raw):
            labels.append(raw)
            continue
        # ALL CAPS headings
        if raw.isupper() and len(raw.split()) > 1 and len(raw.split()) < 10:
            labels.append(raw)
            continue
        # Title Case headings
        if re.match(r"^[A-Z][a-z]+(\s[A-Z][a-z]+)*$", raw) and len(raw.split()) >= 1:
            labels.append(raw)
            continue
    labels = list(dict.fromkeys(labels))
    print(f"Detected {len(labels)} structural headings")
    return labels
headings = extract_headings_general(full_text)

def map_clauses_to_headings(clauses, text, headings):
    """Assign each clause to nearest preceding heading in document."""
    heading_positions = []
    for h in headings:
        pos = text.find(h)
        if pos != -1:
            heading_positions.append((h, pos))
    heading_positions.sort(key=lambda x: x[1])
    if not heading_positions:
        return ["UNLABELED"] * len(clauses)
    assigned = []
    for cl in clauses:
        pos = text.find(cl[:50])  # approximate location lookup
        best = None
        for h, h_pos in heading_positions:
            if h_pos <= pos:
                best = h
            else:
                break
        if best is None:
            best = "UNLABELED"
        assigned.append(best)
    return assigned
labels_struct = map_clauses_to_headings(clauses, full_text, headings)

# If everything is UNLABELED or only one label, we will cluster instead
unique_labels_struct = set(labels_struct)
print(f" Structural label candidates: {len(unique_labels_struct)}")
if len(unique_labels_struct) <= 1:
    print(" Not enough structural labels, switching to semantic clustering for labels...")
    embedder_cluster = SentenceTransformer("all-MiniLM-L6-v2")
    clause_embs = embedder_cluster.encode(clauses)
    # heuristic for number of clusters
    k = min(8, max(2, len(clauses) // 20))
    print(f"Using {k} clusters for semantic labels")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(clause_embs)
    labels = [f"Cluster_{cid}" for cid in cluster_ids]
else:
    labels = labels_struct
print(f"Final label set size: {len(set(labels))}")


def clean_labels_for_stratification(labels):
    labels = list(labels)
    while True:
        counts = Counter(labels)
        rare = [lbl for lbl, cnt in counts.items() if cnt < 2]
        # If no rare labels, we are safe
        if len(rare) == 0:
            return labels
        # Replace rare labels with OTHER
        labels = ["OTHER" if l in rare else l for l in labels]
# STEP 1 — Clean labels completely
labels_clean = clean_labels_for_stratification(labels)
# STEP 2 — Verify no rare labels remain
print("Label counts after cleaning:")
print(Counter(labels_clean))
# FIRST SPLIT — stratified
clauses_train, clauses_temp, labels_train, labels_temp = train_test_split(
    clauses,
    labels_clean,      # use cleaned labels from earlier
    test_size=0.30,
    random_state=42,
    stratify=labels_clean
)
# SECOND SPLIT — NOT stratified (avoid 1-sample classes)
clauses_val, clauses_test, labels_val, labels_test = train_test_split(
    clauses_temp,
    labels_temp,
    test_size=0.50,
    random_state=42,
    shuffle=True       # safe; no stratify
)
print("TRAIN size:", len(clauses_train))
print("VAL size:", len(clauses_val))
print("TEST size:", len(clauses_test))
print("Creating TF-IDF representations for clauses...")
vectorizer_clauses = TfidfVectorizer(max_features=15000, stop_words='english')
X_train = vectorizer_clauses.fit_transform(clauses_train)
X_val = vectorizer_clauses.transform(clauses_val)
X_test = vectorizer_clauses.transform(clauses_test)
print(f"TF-IDF matrix created with shape: {X_train.shape}")

le = LabelEncoder()
y_train = le.fit_transform(labels_train)
y_val = le.transform(labels_val)
y_test = le.transform(labels_test)
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)
params = {
    "objective": "multi:softprob",
    "num_class": len(np.unique(y_train)),
    "eval_metric": "mlogloss",
    "max_depth": 6,
    "eta": 0.15,
    "subsample": 0.9,
}

watchlist = [(dtrain, "train"), (dval, "validation")]
xgb_model = xgb.train(params, dtrain, num_boost_round=500, evals=watchlist, early_stopping_rounds=30)
print("XGBoost trained with early stopping!")


preds = xgb_model.predict(dtest)
pred_labels = np.argmax(preds, axis=1)
acc = accuracy_score(y_test, pred_labels)
prec = precision_score(y_test, pred_labels, average="weighted", zero_division=0)
rec = recall_score(y_test, pred_labels, average="weighted", zero_division=0)
f1 = f1_score(y_test, pred_labels, average="weighted", zero_division=0)

print("\nXGBoost TEST METRICS:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")

def explode_sentences_into_snippets(sentences, min_words=4):
    new_units = []
    for sent in sentences:
        parts = re.split(r'[.;:]', sent)
        for p in parts:
            p = p.strip()
            if len(p.split()) >= min_words:
                new_units.append(p)
    print(f"🔹 Expanded {len(sentences)} clauses → {len(new_units)} snippets")
    return new_units
snippets = explode_sentences_into_snippets(clauses)

from rank_bm25 import BM25Okapi

# TF-IDF for snippets
vectorizer_snip = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1,3))
X_tfidf_snip = vectorizer_snip.fit_transform(snippets)
# BM25
tokenized_snips = [nltk.word_tokenize(s.lower()) for s in snippets]
bm25 = BM25Okapi(tokenized_snips)
# Embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")
emb_matrix_snip = embedder.encode(snippets, convert_to_tensor=True)
# Also map each snippet back to its parent clause index to re-use XGBoost labels
snippet_to_clause_idx = []
for i, sent in enumerate(snippets):
    # naive mapping: find first clause containing this snippet
    parent_idx = 0
    for j, c in enumerate(clauses):
        if sent in c:
            parent_idx = j
            break
    snippet_to_clause_idx.append(parent_idx)

def normalize_scores(scores):
    scores = np.array(scores)
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

def highlight(text, query):
    q_words = set(query.lower().split())
    words = text.split()
    return " ".join([f"**{w}**" if w.lower() in q_words else w for w in words])

def hybrid_xgb_search(question, top_k=5,
                      w_bm25=0.3, w_tfidf=0.25, w_sem=0.25, w_xgb=0.20):
    """
    Hybrid retriever:
    - BM25 over snippets
    - TF-IDF cosine similarity
    - MiniLM semantic similarity
    - XGBoost label prior (clause-level label)
    Returns: list of snippet indices (len <= top_k)
    """
    # --- BM25 ---
    bm25_scores = bm25.get_scores(nltk.word_tokenize(question.lower()))
    bm25_norm = normalize_scores(bm25_scores)
    # --- TF-IDF ---
    q_tfidf = vectorizer_snip.transform([question])
    tfidf_scores = cosine_similarity(q_tfidf, X_tfidf_snip).flatten()
    tfidf_norm = normalize_scores(tfidf_scores)
    # --- Semantic embeddings ---
    q_emb = embedder.encode([question], convert_to_tensor=True)
    sem_scores = cosine_similarity(
        q_emb.cpu().numpy(), emb_matrix_snip.cpu().numpy()
    ).flatten()
    sem_norm = normalize_scores(sem_scores)
    # --- XGBoost label prior (question → label, match snippet’s parent clause label) ---
    q_vec_clause_space = vectorizer_clauses.transform([question])
    q_d = xgb.DMatrix(q_vec_clause_space)
    q_label_proba = xgb_model.predict(q_d)[0]
    q_label_idx = int(np.argmax(q_label_proba))
    target_label = le.inverse_transform([q_label_idx])[0]
    xgb_snip_scores = []
    for idx in snippet_to_clause_idx:
        clause_label = labels[idx]
        score = 1.0 if clause_label == target_label else 0.0
        xgb_snip_scores.append(score)
    xgb_snip_scores = np.array(xgb_snip_scores)
    xgb_norm = normalize_scores(xgb_snip_scores)

    # --- Final hybrid score ---
    final_scores = (
        w_bm25 * bm25_norm +
        w_tfidf * tfidf_norm +
        w_sem * sem_norm +
        w_xgb * xgb_norm
    )
    # top_k indices
    if top_k <= 0:
        return []
    best_idx = final_scores.argsort()[-top_k:][::-1]
    # Debug / inspection printout
    print("\n============================")
    print(f" QUESTION: {question}")
    print("============================\n")
    for rank, idx in enumerate(best_idx, 1):
        print(f" Match {rank} | Score={final_scores[idx]:.3f} | Label={labels[snippet_to_clause_idx[idx]]}")
        print(highlight(snippets[idx], question))
        print("-" * 80)
    #  IMPORTANT: return indices so callers don’t get None
    return list(best_idx)


test_hybrid_qs = [
    "Is my liability limited or unlimited under this agreement?",
    "What are the termination rights?",
    "What is the warranty period?",
    "Are there confidentiality obligations?",
]

for q in test_hybrid_qs:
    hybrid_xgb_search(q, top_k=5)

from sklearn.model_selection import train_test_split

# Build training dataset directly from real clauses (NO synthetic labels)
def make_llm_training_data(clauses):
    data = []
    for c in clauses:
        data.append({
            "instruction": "Rewrite the following legal clause clearly.",
            "input": c,
            "output": c
        })
    return data

train_examples = make_llm_training_data(clauses_train)
val_examples   = make_llm_training_data(clauses_val)
train_dataset = Dataset.from_list(train_examples)
val_dataset   = Dataset.from_list(val_examples)

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import torch, time

print("\n" + "="*60)
print(" MODEL 1: Fine-tuning DistilGPT-2 on legal clauses")
print("="*60)


# Tokenizer & model
model1_name = "distilgpt2"
tokenizer1 = AutoTokenizer.from_pretrained(model1_name)
tokenizer1.pad_token = tokenizer1.eos_token

model1 = AutoModelForCausalLM.from_pretrained(model1_name)

lora_config1 = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["c_attn"]
)
model1 = get_peft_model(model1, lora_config1)

#  Tokenization function
def tokenize_for_gpt2(examples):
    texts = [
        f"### Instruction: {inst}\n### Input: {inp}\n### Output: {out}"
        for inst, inp, out in zip(examples['instruction'], examples['input'], examples['output'])
    ]
    return tokenizer1(texts, truncation=True, padding="max_length", max_length=256)

tokenized_train = train_dataset.map(tokenize_for_gpt2, batched=True, remove_columns=train_dataset.column_names)
tokenized_val   = val_dataset.map(tokenize_for_gpt2, batched=True, remove_columns=val_dataset.column_names)


# Training setup
training_args1 = TrainingArguments(
    output_dir="./distilgpt2-legal-clauses",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    save_steps=100,
    logging_steps=20,
    learning_rate=2e-4,
    fp16=True,
    report_to="none",
    save_total_limit=2,
)

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer1, mlm=False)


trainer = Trainer(
    model=model1,
    args=training_args1,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=collator,
)

trainer.train()
print(" DistilGPT-2 fine-tuning completed!")

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import torch, time

print("\n" + "="*60)
print(" MODEL 2: Fine-tuning FLAN-T5-Small on legal clauses")
print("="*60)

from transformers import T5Tokenizer, T5ForConditionalGeneration

model2_name = "google/flan-t5-small"
tokenizer2 = T5Tokenizer.from_pretrained(model2_name)
model2 = T5ForConditionalGeneration.from_pretrained(model2_name)

lora_config2 = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q", "v"]
)
model2 = get_peft_model(model2, lora_config2)

def tokenize_for_t5(examples):
    inputs = [
        f"Summarize the following legal clause: {inp}"
        for inp in examples['input']
    ]
    model_inputs = tokenizer2(inputs, max_length=256, truncation=True, padding="max_length")

    labels = tokenizer2(
        examples['output'],
        max_length=128,
        truncation=True,
        padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = train_dataset.map(tokenize_for_t5, batched=True, remove_columns=train_dataset.column_names)
tokenized_val   = val_dataset.map(tokenize_for_t5, batched=True, remove_columns=val_dataset.column_names)


training_args2 = TrainingArguments(
    output_dir="./flan-t5-legal-clauses",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    save_steps=100,
    logging_steps=20,
    learning_rate=3e-4,
    fp16=True,
    report_to="none",
    save_total_limit=2,
)

trainer2 = Trainer(
    model=model2,
    args=training_args2,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
)

trainer2.train()
print(" FLAN-T5-Small fine-tuning completed!")

def generate_answer_gpt2(question, top_k=3, max_new_tokens=120):
    snip_idx = hybrid_xgb_search(question, top_k=top_k)
    def safe_snip(text, max_tokens=110):
        ids = tokenizer1.encode(text)
        ids = ids[:max_tokens]
        return tokenizer1.decode(ids)
    context_snips = [safe_snip(snippets[i]) for i in snip_idx]
    context = "\n".join(context_snips)
    # Add "Summary:" because GPT-2 shifts modes
    prompt = (
        "You are a legal assistant. Give a short answer based ONLY on the clauses.\n"
        "Do NOT rewrite or continue the contract. Provide a direct answer.\n\n"
        f"--- CONTRACT EXCERPTS ---\n{context}\n"
        "--- END EXCERPTS ---\n\n"
        f"QUESTION: {question}\n"
        "ANSWER (one sentence): "
    )
    inputs = tokenizer1(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=900
    )
    input_len = inputs["input_ids"].shape[1]
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        model1.to("cuda")
    outputs = model1.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=5,
        do_sample=False,
        no_repeat_ngram_size=3,
        repetition_penalty=1.15,
        early_stopping=True
    )
    gen_tokens = outputs[0][input_len:]
    answer = tokenizer1.decode(gen_tokens, skip_special_tokens=True).strip()
    # Clean out junk
    for m in ["QUESTION:", "CONTRACT", "EXCERPTS", "ANSWER", "---"]:
        idx = answer.find(m)
        if idx > 0:
            answer = answer[:idx].strip()
    # Fallback for hallucinations
    if len(answer.split()) < 2 or any(w in answer.lower() for w in ["adma", "bathes", "payment"]):
        answer = "Not clearly specified in the provided clauses."
    return answer

test_questions = [
    "Is liability capped or uncapped for indirect damages?",
    "What happens if either party terminates the agreement?",
    "Are there any confidentiality obligations?",
    "Is there any warranty and what is its duration?"
]
for q in test_questions:
     ans1 = generate_answer_gpt2(q, top_k=3)
     print(ans1)
print("\n Hybrid legal analysis pipeline finished (XGBoost + 2 fine-tuned LLMs).")

def generate_answer_T5(question, top_k=3, max_new_tokens=120):
    # Retrieve relevant clauses
    snip_idx = hybrid_xgb_search(question, top_k=top_k)
    # Truncate snippets
    def safe_snip(text, max_tokens=110):
        ids = tokenizer2.encode(text)
        ids = ids[:max_tokens]
        return tokenizer2.decode(ids)
    context_snips = [safe_snip(snippets[i]) for i in snip_idx]
    context = "\n".join(context_snips)
    # Build prompt
    prompt = (
        "### Instruction: Using ONLY the following contract clauses, answer the legal question. "
        "If the clauses do not specify an answer, say: 'Not clearly specified.'\n\n"
        f"### Clauses:\n{context}\n\n"
        f"### Question: {question}\n### Answer:"
    )
    # Tokenize
    inputs = tokenizer2(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=900       # keep below 1024 limit
    )
    input_len = inputs["input_ids"].shape[1]
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        model2.to("cuda")
    # Generate only new tokens
    outputs = model2.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=4,
        do_sample=False,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
        early_stopping=True
    )
    # Decode ONLY the generated portion
    gen_tokens = outputs[0][input_len:]
    raw_answer = tokenizer2.decode(gen_tokens, skip_special_tokens=True).strip()
    # Remove leaked prompt text
    cut_markers = ["### Question", "### Clauses", "### Instruction"]
    for m in cut_markers:
        idx = raw_answer.find(m)
        if idx > 0:
            raw_answer = raw_answer[:idx].strip()
    # Fallback for empty or useless answers
    if raw_answer == "" or raw_answer.lower() in ["none", "none.", "yes.", "yes"]:
        raw_answer = "Not clearly specified."
    return raw_answer.strip(). 
test_questions = [
    "Is liability capped or uncapped for indirect damages?",
    "What happens if either party terminates the agreement?",
    "Are there any confidentiality obligations?",
    "Is there any warranty and what is its duration?"
]
for q in test_questions:
     ans1 = generate_answer_T5(q, top_k=3)
     print(ans1)
print("\n Hybrid legal analysis pipeline finished (XGBoost + 2 fine-tuned LLMs).")


# BUILD EVALUATION DATASET FOR LLM
eval_clauses = clauses[:30]   # Take 30 clauses for evaluation
eval_questions = [
    f"Summarize the following legal clause in plain language:\n{c}"
    for c in eval_clauses
]
gold_answers = eval_clauses   # Self-supervised (you trained model to output the clause)
print(" Evaluation set ready:", len(eval_clauses), "samples")

scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
gpt2_preds = []
t5_preds   = []
for clause in eval_clauses:
    gpt2_preds.append(generate_answer_gpt2(clause))
    t5_preds.append(generate_answer_T5(clause))
# ---- METRIC STORAGE ----
gpt2_rouge = []
t5_rouge   = []
gpt2_bleu  = []
t5_bleu    = []
# ROUGE + BLEU
for gold, p1, p2 in zip(gold_answers, gpt2_preds, t5_preds):
    gpt2_rouge.append(scorer.score(gold, p1))
    t5_rouge.append(scorer.score(gold, p2))

    gpt2_bleu.append(sentence_bleu([gold.split()], p1.split()))
    t5_bleu.append(sentence_bleu([gold.split()], p2.split()))
# BERTScore
P1, R1, F1_gpt2 = bertscore(gpt2_preds, gold_answers, lang="en")
P2, R2, F1_t5   = bertscore(t5_preds, gold_answers, lang="en")

print("\n============== GPT-2 SCORES ==============")
print("ROUGE-1:", np.mean([r['rouge1'].fmeasure for r in gpt2_rouge]))
print("ROUGE-2:", np.mean([r['rouge2'].fmeasure for r in gpt2_rouge]))
print("ROUGE-L:", np.mean([r['rougeL'].fmeasure for r in gpt2_rouge]))
print("BLEU:",    np.mean(gpt2_bleu))
print("BERTScore F1:", float(F1_gpt2.mean()))

print("\n============== FLAN-T5 SCORES ==============")
print("ROUGE-1:", np.mean([r['rouge1'].fmeasure for r in t5_rouge]))
print("ROUGE-2:", np.mean([r['rouge2'].fmeasure for r in t5_rouge]))
print("ROUGE-L:", np.mean([r['rougeL'].fmeasure for r in t5_rouge]))
print("BLEU:",    np.mean(t5_bleu))
print("BERTScore F1:", float(F1_t5.mean()))
