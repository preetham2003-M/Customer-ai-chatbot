import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

st.title("Customer Experience AI Chatbot")

# Load dataset
data = pd.read_excel("Customer Experience.xlsx")

# Reduce dataset for cloud performance
data = data.head(2000)

# Convert rows to text
data["text"] = data.astype(str).agg(" ".join, axis=1)

texts = data["text"].tolist()

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings
embeddings = embed_model.encode(texts)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Load HuggingFace LLM
generator = pipeline("text-generation", model="distilgpt2")

query = st.text_input("Ask anything about customer cases")

if query:

    # Convert question to embedding
    query_embedding = embed_model.encode([query])

    # Search similar records
    D, I = index.search(np.array(query_embedding), k=5)

    context = " ".join([texts[i] for i in I[0]])

    prompt = f"""
    Based on the following customer experience data:
    {context}

    Answer the question: {query}
    """

    result = generator(prompt, max_length=150, num_return_sequences=1)

    st.write(result[0]["generated_text"])
