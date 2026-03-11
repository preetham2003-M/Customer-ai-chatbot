import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

st.title("Customer Experience AI Chatbot")

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_excel("Customer Experience.xlsx")
    data = data.head(2000)   # limit rows for cloud performance
    data["text"] = data.astype(str).agg(" ".join, axis=1)
    return data

data = load_data()

texts = data["text"].tolist()

# Load embedding model
@st.cache_resource
def load_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

model = load_model()

# Create FAISS index
@st.cache_resource
def create_index():
    embeddings = model.encode(texts)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

index = create_index()

# User question input
query = st.text_input("Ask anything about customer cases")

if query:

    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=5)

    results = data.iloc[I[0]]

    st.subheader("AI Explanation")

    explanation = f"""
Based on your question **"{query}"**, the chatbot searched the dataset for the most relevant customer cases.

The table below shows customer case records that are closely related to your query.
From these results you can observe information such as:

• Customer case number  
• SLA status  
• Channel used by the customer  
• Category of issue  
• Case status  

These records help identify patterns in customer experience issues.
"""

    st.write(explanation)

    st.subheader("Relevant Data")

    st.dataframe(results)
