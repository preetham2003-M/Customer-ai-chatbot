import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

st.title("Customer Experience AI Chatbot")

@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    generator = pipeline("text-generation", model="distilgpt2")
    return embed_model, generator

@st.cache_data
def load_data():
    data = pd.read_excel("Customer Experience.xlsx")
    data = data.head(2000)
    data["text"] = data.astype(str).agg(" ".join, axis=1)
    return data

embed_model, generator = load_models()
data = load_data()

texts = data["text"].tolist()

embeddings = embed_model.encode(texts)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

query = st.text_input("Ask anything about customer cases")

if query:
    query_embedding = embed_model.encode([query])
    D, I = index.search(np.array(query_embedding), k=5)
    context = " ".join([texts[i] for i in I[0]])

    prompt = f"""
    Based on the following customer data:
    {context}

    Answer the question: {query}
    """

    result = generator(prompt, max_length=150)
    st.write(result[0]["generated_text"])
