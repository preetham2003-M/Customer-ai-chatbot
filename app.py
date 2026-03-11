import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

st.title("Customer Experience AI Chatbot")

@st.cache_resource
def load_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

@st.cache_data
def load_data():
    data = pd.read_excel("Customer Experience.xlsx")
    data = data.head(2000)
    data["text"] = data.astype(str).agg(" ".join, axis=1)
    return data

model = load_model()
data = load_data()

texts = data["text"].tolist()

@st.cache_resource
def create_index():
    embeddings = model.encode(texts)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

index = create_index()

query = st.text_input("Ask anything about customer cases")

if query:
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=5)
    results = data.iloc[I[0]]

    st.write("Relevant data:")
    st.write(results)
    query = st.text_input("Ask anything about customer cases")

if query:

    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=5)

    results = data.iloc[I[0]]

    st.subheader("AI Explanation")

    explanation = f"""
Based on the dataset, the chatbot searched for records related to your question: **{query}**.

The table below shows the most relevant customer cases. 
You can observe the channel, category, status, and type of customer requests from these results.

These records help understand patterns in customer experience issues.
"""

    st.write(explanation)

    st.subheader("Relevant Data")

    st.dataframe(results)

