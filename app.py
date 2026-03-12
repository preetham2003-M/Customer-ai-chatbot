
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

st.title("Customer Experience AI Chatbot")

# -------------------------------
# Load dataset
# -------------------------------
@st.cache_data
def load_data():
    data = pd.read_excel("Book1.xlsx")
    data = data.head(2000)  # limit rows for faster processing
    data["text"] = data.astype(str).agg(" ".join, axis=1)
    return data

data = load_data()
texts = data["text"].tolist()

# -------------------------------
# Load embedding model
# -------------------------------
@st.cache_resource
def load_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

model = load_model()

# -------------------------------
# Create vector index
# -------------------------------
@st.cache_resource
def create_index():
    embeddings = model.encode(texts)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

index = create_index()

# -------------------------------
# User input
# -------------------------------
query = st.text_input("Ask anything about customer cases")

# -------------------------------
# Process query
# -------------------------------
if query:

    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=10)

    results = data.iloc[I[0]]

    # -------------------------------
    # Data analysis for explanation
    # -------------------------------
    channel_counts = results["Channel"].value_counts()
    category_counts = results["Category"].value_counts()
    status_counts = results["Status"].value_counts()

    top_channel = channel_counts.idxmax() if not channel_counts.empty else "Unknown"
    top_category = category_counts.idxmax() if not category_counts.empty else "Unknown"
    top_status = status_counts.idxmax() if not status_counts.empty else "Unknown"

    # -------------------------------
    # AI Explanation
    # -------------------------------
    st.subheader("AI Explanation")

    explanation = f"""
Your question: **{query}**

After analyzing the most relevant customer case records from the dataset, the chatbot identified the following insights:

• The most common **communication channel** among these cases is **{top_channel}**.

• The most frequent **issue category** is **{top_category}**.

• The most common **case status** is **{top_status}**.

These observations are based on the customer cases that are most closely related to your query.

The table below shows the relevant customer case records that were used for this analysis.
"""

    st.write(explanation)

    # -------------------------------
    # Display table
    # -------------------------------
    st.subheader("Relevant Data")

    st.dataframe(results)

