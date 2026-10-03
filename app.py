import streamlit as st
import numpy as np
import pandas as pd
from model_base import ApplicantScreeningNN, min_max_normalize
from resume_parser import parse_resume
import os

st.set_page_config(page_title="AI Recruitment Simulator", layout="wide")

@st.cache_resource
def load_trained_model():
    model_path = 'model_weights.npz'
    if not os.path.exists(model_path):
        st.error(f"Model file {model_path} not found! Please run train.py first.")
        return None
    
    # Initialize dummy and load
    nn = ApplicantScreeningNN()
    nn.load_model(model_path)
    return nn

def manual_forward_pass(nn, X_raw):
    """Executes a manual forward pass for demonstration and returns metrics."""
    # Ensure 2D array
    X_raw = np.array(X_raw).reshape(1, -1)
    
    # 1. Normalize
    X_norm, _, _ = min_max_normalize(X_raw, nn.min_vals, nn.max_vals)
    
    # 2. Forward Propagation steps
    Z1 = np.dot(X_norm, nn.W1) + nn.b1
    A1 = nn.relu(Z1)
    
    Z2 = np.dot(A1, nn.W2) + nn.b2
    A2 = nn.sigmoid(Z2)
    
    score = float(A2[0][0]) * 100
    decision = "Shortlisted" if score >= 50 else "Rejected"
    
    return {
        "X_norm": X_norm.flatten(),
        "Z1": Z1.flatten(),
        "A1": A1.flatten(),
        "Z2": Z2.flatten(),
        "Fit Score": score,
        "Decision": decision
    }

def main():
    st.title("🚀 AI Recruitment Simulation System")
    st.markdown("A pure NumPy Neural Network built from scratch for applicant screening.")
    
    nn = load_trained_model()
    if nn is None:
        return
        
    tab1, tab2 = st.tabs(["📄 Resume Screening (PDF)", "🎛️ Manual Simulation"])
    
    with tab1:
        st.header("Upload a Resume for AI Screening")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None:
            with st.spinner('Parsing resume...'):
                features = parse_resume(uploaded_file)
                
            st.subheader("Extracted Candidate Features")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Years of Experience", f"{features['Experience']} Yrs")
            col2.metric("Technical Score", f"{features['Tech_Score']}/100")
            col3.metric("Soft Skills Score", f"{features['Soft_Score']}/100")
            col4.metric("Projects Completed", f"{features['Projects']}")
            
            # Forward Pass
            X_raw = [features['Experience'], features['Tech_Score'], features['Soft_Score'], features['Projects']]
            results = manual_forward_pass(nn, X_raw)
            
            st.divider()
            st.subheader("Neural Network Inference (Forward Pass)")
            
            with st.expander("Show Matrix Math (Under the hood)"):
                st.write("**Normalized Input (X):**", np.round(results["X_norm"], 4))
                st.write("**Hidden Layer Pre-activation (Z1 = X.W1 + b1):**", np.round(results["Z1"], 4))
                st.write("**Hidden Layer Activation (A1 = ReLU(Z1)):**", np.round(results["A1"], 4))
                st.write("**Output Layer Pre-activation (Z2 = A1.W2 + b2):**", np.round(results["Z2"], 4))
                
            st.divider()
            st.subheader("Final Decision")
            if results["Decision"] == "Shortlisted":
                st.success(f"🎉 **{results['Decision']}** | Fit Score: {results['Fit Score']:.2f}%")
            else:
                st.error(f"❌ **{results['Decision']}** | Fit Score: {results['Fit Score']:.2f}%")
                
    with tab2:
        st.header("Manual Candidate Simulation")
        
        col_slider1, col_slider2 = st.columns(2)
        with col_slider1:
            exp_val = st.slider("Years of Experience", 0.0, 10.0, 3.0, 0.5)
            tech_val = st.slider("Technical Score", 0, 100, 60, 1)
        with col_slider2:
            soft_val = st.slider("Soft Skills Score", 0, 100, 50, 1)
            proj_val = st.slider("Projects Completed", 0, 6, 2, 1)
            
        X_raw = [exp_val, tech_val, soft_val, proj_val]
        results = manual_forward_pass(nn, X_raw)
        
        st.markdown("### Prediction")
        if results["Decision"] == "Shortlisted":
            st.success(f"Decision: **{results['Decision']}** (Fit Score: {results['Fit Score']:.2f}%)")
        else:
            st.error(f"Decision: **{results['Decision']}** (Fit Score: {results['Fit Score']:.2f}%)")
            
        st.divider()
        st.subheader("Batch Screening Leaderboard (Test Data)")
        
        if os.path.exists('test_candidates.csv'):
            df = pd.read_csv('test_candidates.csv')
            
            # Predict for all
            X_test_raw = df[['years_experience', 'technical_score', 'soft_skills_score', 'projects_completed']].values
            X_norm, _, _ = min_max_normalize(X_test_raw, nn.min_vals, nn.max_vals)
            
            _, probs = nn.predict(X_norm)
            df['Fit Score (%)'] = (probs.flatten() * 100).round(2)
            df['Decision'] = df['Fit Score (%)'].apply(lambda x: "Shortlisted" if x >= 50 else "Rejected")
            
            # Sort by Fit Score
            df_sorted = df.sort_values(by='Fit Score (%)', ascending=False).reset_index(drop=True)
            df_sorted.index = df_sorted.index + 1
            st.dataframe(df_sorted.style.highlight_max(subset=['Fit Score (%)'], color='lightgreen'), use_container_width=True)
        else:
            st.warning("test_candidates.csv not found! Run train.py first.")

if __name__ == '__main__':
    main()
