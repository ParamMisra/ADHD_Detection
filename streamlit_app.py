import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.pred import load_and_merge_data, train_final_model, BEST_FEATURES, TARGET_COL

# setup
st.set_page_config(page_title="ADHD Diagnostic Support", layout="centered")

st.title("ADHD Diagnostic Support")
st.markdown("### Clinical Assessment Analysis")
st.markdown("---")

# Load and train the machine learning model
@st.cache_resource
def load_machine_learning_model():
    dataset_features, dataset_labels = load_and_merge_data(feature_list=BEST_FEATURES)
    trained_model = train_final_model(dataset_features, dataset_labels)
    return trained_model

with st.spinner("Initializing model..."):
    diagnostic_model = load_machine_learning_model()

st.info("""
**Methodology**
- **Objective:** Adjunctive tool for ADHD evaluation complementing qualitative assessment.
- **Model:** Logistic Regression on 75 biomarkers from CPT-II & Demographics.
- **Note:** Not for standalone diagnosis.
""")

st.subheader("Upload Patient Record")
uploaded_csv_file = st.file_uploader("CPT-II CSV Record", type=["csv"])

if not uploaded_csv_file:
    st.write("Please upload a patient CSV file to generate an analysis.")
else:
    patient_data = pd.read_csv(uploaded_csv_file, sep=None, engine="python")
    
    missing_columns = [col for col in BEST_FEATURES if col not in patient_data.columns]
    
    if missing_columns:
        st.error("Invalid file: Missing required feature columns in the uploaded CSV.")
    else:
        first_patient_record = patient_data.iloc[[0]]
        patient_features = first_patient_record[BEST_FEATURES]
        
        adhd_probability = float(diagnostic_model.predict_proba(patient_features)[:, 1][0])
        predicted_class = int(diagnostic_model.predict(patient_features)[0])
        
        st.markdown("---")
        st.subheader("Diagnostic Results")
        
        if predicted_class == 1:
            st.error("ADHD POSITIVE")
        else:
            st.success("ADHD NEGATIVE")
            
        st.metric("Probability of ADHD", f"{adhd_probability:.1%}")
        st.progress(adhd_probability, text="Confidence Assessment (Risk Scale)")
        
        # Ground truth validation
        if TARGET_COL in first_patient_record.columns:
            actual_diagnosis = int(first_patient_record[TARGET_COL].values[0])
            is_correct = (predicted_class == actual_diagnosis)
            ground_truth_label = "Positive" if actual_diagnosis == 1 else "Negative"
            
            st.markdown("---")
            if is_correct:
                st.success(f"Matched Ground Truth ({ground_truth_label})")
            else:
                st.warning(f"Did not match Ground Truth ({ground_truth_label})")
