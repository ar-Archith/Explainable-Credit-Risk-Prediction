# ================================
# IMPORT LIBRARIES
# ================================
import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import os
import google.generativeai as genai
import time
from dotenv import load_dotenv
import plotly.graph_objects as go

# ================================
# FUNCTIONS
# ================================
def plot_risk_gauge(prob):
    # color zones
    if prob < 0.3:
        color = "green"
    elif prob < 0.6:
        color = "orange"
    else:
        color = "red"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        number={'valueformat': '.2f'},
        title={'text': "Default Risk"},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 0.3], 'color': "lightgreen"},
                {'range': [0.3, 0.6], 'color': "khaki"},
                {'range': [0.6, 1], 'color': "lightcoral"},
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 0.5
            }
        }
    ))

    return fig

def apply_domain_rules(df):
    df = df.copy()
    df["person_age"] = df["person_age"].clip(18, 80)
    df["person_emp_length"] = df["person_emp_length"].clip(0, 40)
    df["person_emp_length"] = np.minimum(df["person_emp_length"], df["person_age"] - 18)
    df["cb_person_cred_hist_length"] = df["cb_person_cred_hist_length"].clip(0, 30)
    return df

quantile_bounds = joblib.load("model/quantile_bounds.pkl")

def apply_quantile_bounds(df, bounds):
    df = df.copy()
    for col, (low, high) in bounds.items():
        df[col] = df[col].clip(low, high)
    return df

# CLEAN FEATURE NAMES
def clean_feature_name(name):
    name = name.replace("num__", "").replace("cat__", "")
    name = name.replace("_", " ")
    return name.title()

def generate_reason_tags(input_df, model):
    import shap
    import numpy as np

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    X_processed = preprocessor.transform(input_df)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_processed)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    all_tags = []

    for row_idx in range(X_processed.shape[0]):
        shap_vals = shap_values[row_idx]

        # top 5 important features
        top_idx = np.argsort(np.abs(shap_vals))[-5:][::-1]

        tags = []
        seen = set()

        for i in top_idx:
            val = shap_vals[i]
            fname = feature_names[i]

            # skip noise
            if abs(val) < 0.01:
                continue

            # -------- TAG LOGIC --------
            if "person_income" in fname:
                tag = "high income" if val < 0 else "low income"

            elif "loan_percent_income" in fname:
                tag = "low loan ratio" if val < 0 else "high loan ratio"

            elif "loan_grade" in fname:
                grade = fname.split("_")[-1]
                tag = f"grade {grade}"

            elif "person_home_ownership" in fname:
                home = fname.split("_")[-1]
                tag = home.lower()

            elif "loan_intent" in fname:
                intent = fname.split("_")[-1]
                tag = intent.lower()

            elif "cb_person_default_on_file" in fname:
                tag = "previous default" if "Y" in fname else "no default"

            elif "loan_int_rate" in fname:
                tag = "high interest" if val > 0 else "low interest"

            elif "person_emp_length" in fname:
                tag = "short employment" if val > 0 else "stable employment"

            else:
                continue

            if tag not in seen:
                seen.add(tag)
                tags.append(tag)

        all_tags.append(", ".join(tags[:5]))

    return all_tags

def batch_prediction_module(model, quantile_bounds):
    import pandas as pd
    import streamlit as st

    st.subheader("📂 Batch Prediction (Upload CSV)")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)

            st.write("### Preview of Uploaded Data")
            st.dataframe(df_batch.head())

            # 🔍 Check required columns
            required_cols = [
                "person_age", "person_income", "person_home_ownership",
                "person_emp_length", "loan_intent", "loan_grade",
                "loan_amnt", "loan_int_rate", "loan_percent_income",
                "cb_person_default_on_file", "cb_person_cred_hist_length"
            ]

            missing_cols = [col for col in required_cols if col not in df_batch.columns]

            if missing_cols:
                st.error(f"Missing columns: {missing_cols}")
                return

            # 🧠 Apply preprocessing (same as training)
            df_batch = apply_domain_rules(df_batch)
            df_batch = apply_quantile_bounds(df_batch, quantile_bounds)

            # 🔥 Predict
            probs = model.predict_proba(df_batch)[:, 1]

            df_batch["default_probability"] = probs
            df_batch["decision"] = df_batch["default_probability"].apply(
                lambda x: "Reject" if x > 0.5 else "Approve"
            )

            # 📊 Sort by risk
            df_batch = df_batch.sort_values(by="default_probability", ascending=False)

            st.write("Generating explanations...")
            df_batch["reason_tags"] = generate_reason_tags(df_batch, model)

            st.write("### 📊 Prediction Results")
            st.dataframe(df_batch)

            # 📥 Download button
            csv = df_batch.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name="credit_predictions.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# ================================
# DEBUG
# ================================
st.write("App started successfully")

# ================================
# LOAD MODEL
# ================================
model = joblib.load("model/credit_risk_model.pkl")
# feature_names = joblib.load("model/feature_names.pkl")

# ================================
# GEMINI SETUP
# ================================
load_dotenv()
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
st.markdown("# 💳 AI Credit Risk Predictor")
st.caption("Explainable AI system for loan approval decisions")

mode = st.radio(
    "Select Mode",
    ["Single Prediction", "Batch Prediction (CSV)"],
    horizontal=True
)

st.divider()

if mode == "Single Prediction":
    st.write("Fill the loan details below:")

    # ================================
    # INPUT FIELDS
    # ================================

    col1, col2 = st.columns(2)

    with col1:
        person_age = st.number_input("Age", 18, 120, 25)
        person_income = st.number_input("Income", 1000, 1000000, 50000)
        person_emp_length = st.number_input("Employment Length", 0, 50, 5)

    with col2:
        loan_amnt = st.number_input("Loan Amount", 500, 35000, 10000)
        loan_int_rate = st.number_input("Interest Rate", 5.0, 32.0, 12.0)
        cb_person_cred_hist_length = st.number_input("Credit History Length", 0, 30, 5)

    loan_percent_income = loan_amnt / person_income
    st.write(f"Loan / Income Ratio: {loan_percent_income:.2f}")

    st.subheader("Loan Details")

    col3, col4 = st.columns(2)

    with col3:
        person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
        loan_intent = st.selectbox("Loan Intent", [
                                        "PERSONAL",
                                        "EDUCATION",
                                        "MEDICAL",
                                        "VENTURE",
                                        "HOMEIMPROVEMENT",
                                        "DEBTCONSOLIDATION"
                                    ]
                                )

    with col4:
        loan_grade = st.selectbox("Loan Grade", ["A","B","C","D","E","F","G"])
        cb_person_default_on_file = st.selectbox("Previous Default", ["Y", "N"])

    st.divider()

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

        input_df = apply_domain_rules(input_df)
        input_df = apply_quantile_bounds(input_df, quantile_bounds)

        prob = model.predict_proba(input_df)[0][1]

        # Save to session
        st.session_state["input_df"] = input_df
        st.session_state["prob"] = prob

        st.divider()

    # ================================
    # SHOW RESULTS (OUTSIDE BUTTON)
    # ================================
    if "prob" in st.session_state:

        prob = st.session_state["prob"]
        input_df = st.session_state["input_df"]

        st.subheader("📊 Prediction Result")
        # st.write(f"Default Probability: **{prob:.2f}**")
        if prob < 0.3:
            risk_label = "Low Risk"
        elif prob < 0.6:
            risk_label = "Moderate Risk"
        else:
            risk_label = "High Risk"

        st.markdown(f"### Risk Level: **{risk_label}**")

        if prob > 0.5:
            st.error("🚫 High Risk — Loan Not Recommended")
        else:
            st.success("✅ Low Risk — Loan Can Be Approved")
        
        fig = plot_risk_gauge(prob)
        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()

        # ================================
        # SHAP EXPLANATION
        # ================================

        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]

        X_processed = preprocessor.transform(input_df)

        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_processed)

        actual_features = preprocessor.get_feature_names_out()

        # debug_df = pd.DataFrame(
        #     X_processed,
        #     columns=actual_features
        # )

        # st.write("Processed Input (Debug):")
        # st.dataframe(debug_df.T)

        # Handle list output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # SHAP PROCESSING
        shap_vals = shap_values[0]

        # Top 5 features by importance
        top_idx = np.argsort(np.abs(shap_vals))[-5:][::-1]

        unique_explanations = []
        seen = set()
        candidate_idx = np.argsort(np.abs(shap_vals))[-10:][::-1]

        for i in candidate_idx:
            feature = actual_features[i]
            value = shap_vals[i]

            # skip tiny impacts (noise)
            if abs(value) < 0.01:
                continue

            # -------- HANDLE CATEGORICAL FEATURES --------
            if "person_home_ownership" in feature:
                name = f"Home Ownership = {person_home_ownership}"
            elif "loan_intent" in feature:
                name = f"Loan Intent = {loan_intent}"
            elif "loan_grade" in feature:
                name = f"Loan Grade = {loan_grade}"
            elif "cb_person_default_on_file" in feature:
                name = f"Previous Default = {cb_person_default_on_file}"
            else:
                name = clean_feature_name(feature)

            impact = "increased default risk" if value > 0 else "reduced default risk"
            text = f"{name} → {impact}"

            # remove duplicates
            if text not in seen:
                seen.add(text)
                unique_explanations.append(text)
            
            # keep only 5
            if len(unique_explanations) == 5:
                break

        # Display nicely
        st.subheader("🔍 Key Factors Influencing Decision")
        for exp in unique_explanations:
            st.markdown(f"• {exp}")

        # Prepare for Gemini
        explanation_text = "\n".join(unique_explanations)

        st.divider()

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
                    You are a professional credit risk analyst.
                    Customer Profile:
                    - Age: {person_age}
                    - Income: {person_income}
                    - Employment Length: {person_emp_length}
                    - Loan Amount: {loan_amnt}
                    - Interest Rate: {loan_int_rate}
                    - Loan Intent: {loan_intent}
                    - Loan Grade: {loan_grade}
                    - Credit History Length: {cb_person_cred_hist_length}
                    - Previous Default: {cb_person_default_on_file}
                    Model Prediction:
                    - Default Probability: {prob:.2f}
                    Key Risk Drivers:
                    {explanation_text}
                    Task:
                    Explain clearly and concisely:
                    1. WHY the model made this decision
                    2. Highlight key risks and how the customer can improve their profile
                    3. Justify the SAME decision (do NOT contradict)
                    Keep it simple, professional, and under 5 lines.
                    """

                    try:
                        with st.spinner("Generating AI explanation..."):
                            explanation = get_ai_explanation(prompt)
                            st.write(explanation)
                    except Exception:
                        st.warning("Rate limit hit. Try again.")
else:
    batch_prediction_module(model, quantile_bounds)