import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

# --- Page Configuration ---
st.set_page_config(
    page_title="RenoMed CKD Prediction - Google Health",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
    <style>
        .main-title {
            font-size: 40px;
            font-weight: bold;
            color: #4285F4;
            text-align: center;
        }
        .subtitle {
            font-size: 22px;
            color: #34A853;
            text-align: center;
        }
        .footer {
            text-align: center;
            margin-top: 50px;
            color: gray;
        }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown(
    """<div style='display: flex; align-items: center; justify-content: center;'>
        <img src='https://th.bing.com/th/id/R.ca2dcc1ba94081919554b36010545de7?rik=f5wZ%2b3u2QDlV7Q&riu=http%3a%2f%2f1.bp.blogspot.com%2f_nIst9zD6rnM%2fSo-kLSQGPhI%2fAAAAAAAABH4%2fwwWtZ5D5ZzI%2fs400%2fwindowslivewritercomingsoongooglehealthrecords-f42logo-googlehealth.jpg43.gif&ehk=%2b3JpA09TIYpNestw%2faPtiXbMCGyZVET9cFhCVW6WXu0%3d&risl=&pid=ImgRaw&r=0' width='200' style='margin-right: 20px;'>
        <span class='main-title'>RenoMed CKD Prediction</span>
    </div>""",
    unsafe_allow_html=True
)
st.markdown("<p class='subtitle'>Google Health Internal Application</p>", unsafe_allow_html=True)
st.write("---")

# --- Project Overview ---
st.header(" Project Overview")
st.markdown("""
RenoMed CKD is a predictive application developed under **Google Health** to support early detection of **Chronic Kidney Disease (CKD)**.  
By leveraging patient clinical and laboratory data, this tool empowers healthcare professionals to make timely interventions.

### Key Highlights:
- **Data Preprocessing:** Missing values handled, ensuring consistency.
- **Feature Selection:** Top 10 influential features identified.
- **Model Training:** Random Forest & Logistic Regression models applied.
- **High Accuracy:** Achieved strong cross-validation performance.

> **Mission:** Early CKD detection saves lives. RenoMed CKD bridges data science and clinical practice.
""")
st.write("---")

# --- Load Models ---
@st.cache_resource
def load_model(model_name):
    try:
        with open(model_name, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error(f"Model file '{model_name}' not found.")
        return None

rf_model = load_model('random_forest_model.pkl')
lr_model = load_model('logistic_regression_model.pkl')

# --- Prediction Function ---
def predict_ckd(model, input_data):
    features = ['sg', 'al', 'pc', 'bu', 'sc', 'hemo', 'pcv', 'rc', 'htn', 'dm']
    input_df = pd.DataFrame([input_data], columns=features)
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[:, 1][0]
    return prediction[0], probability

# --- Sidebar ---
st.sidebar.header("Model Selection")
selected_model_name = st.sidebar.radio(
    "Choose a Prediction Model:",
    ('Random Forest Classifier', 'Logistic Regression')
)
selected_model = rf_model if selected_model_name == 'Random Forest Classifier' else lr_model

# --- Input Form ---
st.header(" Predict Chronic Kidney Disease")
st.write("Enter patient clinical and laboratory data:")

prediction = None
probability = None
input_data = None

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        sg = st.number_input('Specific Gravity (sg)', 1.000, 1.030, 1.017, 0.001, format="%.3f")
        al = st.number_input('Albumin (al)', 0.0, 5.0, 1.0, 1.0, format="%.1f")
        bu = st.number_input('Blood Urea (bu)', 1.0, 300.0, 65.0, 1.0)
        sc = st.number_input('Serum Creatinine (sc)', 0.1, 50.0, 3.5, 0.1, format="%.1f")
        hemo = st.number_input('Hemoglobin (hemo)', 5.0, 20.0, 12.5, 0.1, format="%.1f")
    with col2:
        pc = 1 if st.selectbox('Pus Cell (pc)', ['Normal', 'Abnormal']) == 'Normal' else 0
        pcv = st.number_input('Packed Cell Volume (pcv)', 10.0, 60.0, 39.0, 1.0)
        rc = st.number_input('Red Blood Cell Count (rc)', 0.0, 10.0, 4.5, 0.1, format="%.1f")
        htn = 1 if st.selectbox('Hypertension (htn)', ['No', 'Yes']) == 'Yes' else 0
        dm = 1 if st.selectbox('Diabetes Mellitus (dm)', ['No', 'Yes']) == 'Yes' else 0

    submitted = st.form_submit_button(" Predict CKD", use_container_width=True)

    if submitted and selected_model is not None:
        input_data = {'sg': sg, 'al': al, 'pc': pc, 'bu': bu, 'sc': sc,
                      'hemo': hemo, 'pcv': pcv, 'rc': rc, 'htn': htn, 'dm': dm}
        prediction, probability = predict_ckd(selected_model, input_data)

# --- Results & Visualizations ---
if prediction is not None:
    st.write("---")
    st.subheader(" Prediction Result")
    if prediction == 1:
        st.success(f" The model predicts: **Chronic Kidney Disease (CKD)**")
        st.info(f"Probability of CKD: {probability:.2f}")
    else:
        st.success(f"🟢 The model predicts: **No Chronic Kidney Disease**")
        st.info(f"Probability of Not CKD: {1 - probability:.2f}")

    # --- Feature Importance ---
    if selected_model_name == 'Random Forest Classifier' and rf_model is not None:
        st.subheader(" Feature Importance")
        try:
            importances = rf_model.feature_importances_
            features = ['sg', 'al', 'pc', 'bu', 'sc', 'hemo', 'pcv', 'rc', 'htn', 'dm']
            importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
            importance_df = importance_df.sort_values(by='Importance', ascending=False)

            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x='Importance', y='Feature', data=importance_df, palette="Blues_r", ax=ax)
            ax.set_title("Top Features Driving CKD Prediction")
            st.pyplot(fig)
        except Exception:
            st.warning("Feature importance not available for this model.")

    # --- Probability Bar ---
    st.subheader(" Probability Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(['Not CKD', 'CKD'], [1 - probability, probability], color=['#34A853', '#EA4335'])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Probability")
    st.pyplot(fig)

    # --- Patient Input Summary ---
    st.subheader(" Patient Input Summary")
    st.table(pd.DataFrame([input_data]))

    # --- Download Report (CSV) ---
    csv_data = pd.DataFrame([input_data]).assign(Prediction=prediction, Probability=probability).to_csv(index=False)
    st.download_button(
        label="Download Report (CSV)",
        data=csv_data,
        file_name="RenoMed_CKD_Report.csv",
        mime="text/csv"
    )

    # --- Download Report (PDF) ---
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="RenoMed CKD Prediction Report", ln=True, align="C")
    pdf.ln(10)
    for key, value in input_data.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)
    pdf.cell(200, 10, txt=f"Prediction: {'CKD' if prediction == 1 else 'Not CKD'}", ln=True)
    pdf.cell(200, 10, txt=f"Probability: {probability:.2f}", ln=True)
    pdf_output = pdf.output(dest="S").encode("latin-1")

    st.download_button(
        label="Download Report (PDF)",
        data=pdf_output,
        file_name="RenoMed_CKD_Report.pdf",
        mime="application/pdf"
    )

# --- Footer ---
st.markdown("""
<div class='footer'>
    <p>Designed by:</p>
    <h3>Gabriel Agana Anongwin</h3>
    <p>Doctor of Pharmacy, Class of 2026</p>
    <p>Data Science Student, ALX Cohort 10</p>
</div>
""", unsafe_allow_html=True)
