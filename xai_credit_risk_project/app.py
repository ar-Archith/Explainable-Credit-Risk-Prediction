import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import os
import google.generativeai as genai
import time

# ================================
# DEBUG
# ================================
st.write("App started successfully")

# ================================
# LOAD MODEL
# ================================
model = joblib.load("model/credit_risk_model.pkl")
feature_names = joblib.load("model/feature_names.pkl")

# ================================
# GEMINI SETUP
# ================================
api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
    llm = genai.GenerativeModel("gemini-2.5-flash-lite")
else:
    llm = None
    st.warning("Gemini API key not found")

# ================================
# CACHE AI RESPONSE (IMPORTANT)
# ================================
@st.cache_data(show_spinner=False)
def get_ai_explanation(prompt):
    return llm.generate_content(prompt).text

# ================================
# UI
# ================================
st.title("💳 AI Credit Risk Predictor")
st.write("Fill the loan details below:")

# ================================
# INPUT FIELDS
# ================================
person_age = st.number_input("Age", 18, 100, 25)
person_income = st.number_input("Income", 1000, 1000000, 50000)
person_home_ownership = st.selectbox(
    "Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"]
)
person_emp_length = st.number_input("Employment Length", 0, 50, 5)

loan_intent = st.selectbox(
    "Loan Intent",
    [
        "PERSONAL",
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "HOMEIMPROVEMENT",
        "DEBTCONSOLIDATION",
    ],
)

loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
loan_amnt = st.number_input("Loan Amount", 500, 100000, 10000)
loan_int_rate = st.number_input("Interest Rate", 1.0, 30.0, 12.0)

loan_percent_income = st.number_input("Loan / Income Ratio", 0.0, 1.0, 0.2)
cb_person_default_on_file = st.selectbox("Previous Default", ["Y", "N"])
cb_person_cred_hist_length = st.number_input("Credit History Length", 1, 30, 5)

# ================================
# PREDICT BUTTON
# ================================
if st.button("Predict Risk"):

    input_df = pd.DataFrame({
        "person_age": [person_age],
        "person_income": [person_income],
        "person_home_ownership": [person_home_ownership],
        "person_emp_length": [person_emp_length],
        "loan_intent": [loan_intent],
        "loan_grade": [loan_grade],
        "loan_amnt": [loan_amnt],
        "loan_int_rate": [loan_int_rate],
        "loan_percent_income": [loan_percent_income],
        "cb_person_default_on_file": [cb_person_default_on_file],
        "cb_person_cred_hist_length": [cb_person_cred_hist_length],
    })

    prob = model.predict_proba(input_df)[0][1]

    # Save to session
    st.session_state["input_df"] = input_df
    st.session_state["prob"] = prob

# ================================
# SHOW RESULTS (OUTSIDE BUTTON)
# ================================
if "prob" in st.session_state:

    prob = st.session_state["prob"]
    input_df = st.session_state["input_df"]

    st.subheader("📊 Prediction Result")
    st.write(f"Default Probability: **{prob:.2f}**")

    if prob > 0.5:
        st.error("🚫 Loan Rejected (High Risk)")
    else:
        st.success("✅ Loan Approved (Low Risk)")

    # ================================
    # SHAP EXPLANATION
    # ================================
    st.subheader("🔍 Model Explanation")

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    X_processed = preprocessor.transform(input_df)

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_processed)

    # FIX: handle list output
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_vals = np.abs(shap_values[0])
    top_idx = np.argsort(shap_vals)[-5:]

    explanation_text = ""
    for i in top_idx:
        explanation_text += f"{feature_names[i]} (impact: {shap_vals[i]:.3f})\n"

    st.text("Top contributing factors:\n" + explanation_text)

    # ================================
    # GEMINI EXPLANATION
    # ================================
    st.subheader("🤖 AI Explanation")

    if st.button("Explain with AI"):

        if llm is None:
            st.warning("Gemini API not configured")
        else:
            last_call_time = st.session_state.get("last_call_time", 0)

            if time.time() - last_call_time < 10:
                st.warning("Wait a few seconds before next request")
            else:
                st.session_state["last_call_time"] = time.time()

                prompt = f"""
                You are a financial risk analyst.

                Default probability: {prob:.2f}

                Key factors:
                {explanation_text}

                Explain briefly (max 4 lines):
                - Why approved/rejected
                - Key risks
                - Final recommendation
                """

                try:
                    explanation = get_ai_explanation(prompt)
                    st.write(explanation)
                except Exception:
                    st.warning("Rate limit hit. Try again.")