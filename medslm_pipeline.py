import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = 'data/med_abstracts.json'
PROCESSED_PATH = 'data/processed_abstracts.json'
EMBEDDINGS_PATH = 'data/embeddings.npy'
INDEX_PATH = 'data/faiss.index'

def preprocess_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found.")

    with open(DATA_PATH, 'r') as f:
        raw_data = json.load(f)

    processed = []
    for entry in raw_data:
        sections = [entry.get(sec, '') for sec in ['background', 'objective', 'methods', 'results', 'conclusions']]
        full_text = ' '.join(filter(None, sections)).strip()
        if full_text:
            processed.append({"pmid": entry.get("pmid", ""), "text": full_text})

    os.makedirs("data", exist_ok=True)
    with open(PROCESSED_PATH, 'w') as f:
        json.dump(processed, f, indent=2)

    print(f"✅ Preprocessed {len(processed)} abstracts")
    return processed

def generate_embeddings():
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    with open(PROCESSED_PATH, 'r') as f:
        processed = json.load(f)

    texts = [doc["text"] for doc in processed]
    embeddings = model.encode(texts, show_progress_bar=True)
    np.save(EMBEDDINGS_PATH, embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)

    print(f"✅ Generated embeddings and FAISS index for {len(texts)} documents")

def retrieve_and_generate_answer(query):
    if not os.path.exists(INDEX_PATH):
        print("Building index...")
        preprocess_data()
        generate_embeddings()

    index = faiss.read_index(INDEX_PATH)
    with open(PROCESSED_PATH, 'r') as f:
        docs = json.load(f)

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    query_vector = model.encode([query])
    distances, indices = index.search(query_vector, 5)
    retrieved_docs = [docs[i]["text"] for i in indices[0] if i < len(docs)]

    context = "\n\n".join(retrieved_docs)[:1500]
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    client = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a medical research assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=500,
    )

    answer = response.choices[0].message.content.strip()
    print(f"\n💬 Answer:\n{answer}\n")

    os.makedirs("data", exist_ok=True)
    with open("data/generated_answers.json", "w") as f:
        json.dump({"query": query, "response": answer}, f, indent=2)

    return answer

if __name__ == "__main__":
    print("🩺 Running MedSLM Pipeline (No Airflow)")
    query = input("Enter your medical query: ")
    retrieve_and_generate_answer(query)
