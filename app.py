import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load dataset
data = pd.read_excel("Customer Experience.xlsx", engine="openpyxl")

# Convert dataset rows into text
data["text"] = data.astype(str).agg(" ".join, axis=1)

# Load HuggingFace embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert dataset into embeddings
embeddings = model.encode(data["text"].tolist())

# Create FAISS vector database
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Streamlit interface
st.title("Customer Experience AI Chatbot")

query = st.text_input("Ask a question about customer cases")

if query:
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=5)
    results = data.iloc[I[0]]

    st.write("Relevant results:")
    st.write(results)