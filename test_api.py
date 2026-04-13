import os
import google.generativeai as genai
import streamlit as st

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
llm = genai.GenerativeModel("gemini-pro")

if st.button("Test Gemini"):
    response = llm.generate_content("Explain credit risk in simple terms")
    st.write(response.text)