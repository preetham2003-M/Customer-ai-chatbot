import streamlit as st
from agent import agent_answer

st.set_page_config(page_title="Customer Experience AI Chatbot", layout="wide")

st.title("Customer Experience AI Chatbot")

st.write("Ask anything about customer cases")

question = st.text_input("Enter your question")

if st.button("Analyze"):

    if question:

        with st.spinner("Analyzing dataset..."):

            result, reasoning = agent_answer(question)

        st.subheader("AI Explanation")
        st.write(reasoning)

        st.subheader("Relevant Data")

        try:
            st.dataframe(result)
        except:
            st.write(result)