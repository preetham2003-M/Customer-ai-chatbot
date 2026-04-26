import google.generativeai as genai
import pandas as pd
import json
import os

# -----------------------------
# Configure Gemini
# -----------------------------
genai.configure(api_key=("AIzaSyAuB51OwU-Ju8dYwo43PTa7x_cejJ-92go"))
model = genai.GenerativeModel("gemini-2.5-flash")

# -----------------------------
# Load Dataset (CHANGED HERE)
# -----------------------------
df = pd.read_excel("Customer Experience.xlsx")

# -----------------------------
# Code Generator
# -----------------------------
def generate_python_code(question):

    prompt = f"""
You are an expert customer data analyst.

Write pandas code to answer the question.

Dataset columns:
{list(df.columns)}

Rules:
1. Use dataframe name df
2. Only use given columns
3. Store final result in variable result
4. Return JSON only

Format:
{{
"python_code": "valid pandas code"
}}

Question:
{question}
"""

    response = model.generate_content(prompt)

    text = response.text.strip()
    text = text.replace("```json","").replace("```","")

    try:
        data = json.loads(text)
        return data["python_code"]
    except:
        return ""


# -----------------------------
# Debugger
# -----------------------------
def debug_python_code(question, code, error):

    prompt = f"""
The python code produced an error.

Question:
{question}

Code:
{code}

Error:
{error}

Fix the code.

Return JSON:
{{
"python_code":"corrected code"
}}
"""

    response = model.generate_content(prompt)

    text = response.text.strip()
    text = text.replace("```json","").replace("```","")

    try:
        data = json.loads(text)
        return data["python_code"]
    except:
        return code


# -----------------------------
# Execute Code
# -----------------------------
def execute_python(code, question):

    safe_globals = {"df": df}

    try:
        exec(code, safe_globals)
        result = safe_globals.get("result", "No result returned")
        return result

    except Exception as e:

        fixed_code = debug_python_code(question, code, str(e))

        try:
            exec(fixed_code, safe_globals)
            result = safe_globals.get("result", "No result returned")
            return result

        except Exception as e2:
            return f"Execution Error: {str(e2)}"


# -----------------------------
# Reasoning Generator (MODIFIED TEXT)
# -----------------------------
def generate_reasoning(question, result):

    prompt = f"""
You are a customer experience analyst.

Explain clearly.

Question:
{question}

Result:
{result}

Dataset Columns:
{list(df.columns)}

Explain in steps:
1. Understanding the question
2. Columns used
3. Data processing
4. Result explanation
5. Final insight
"""

    response = model.generate_content(prompt)
    return response.text


# -----------------------------
# Agent Pipeline
# -----------------------------
def agent_answer(question):

    python_code = generate_python_code(question)
    result = execute_python(python_code, question)
    reasoning = generate_reasoning(question, result)

    return result, reasoning