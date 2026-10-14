"""
AI Recruitment Simulation System - Enhanced Streamlit Dashboard
================================================================
Integrates all 3 custom neural networks:
    Tab 1: Intelligent Candidate Matching (BiLSTM-CRF + Siamese + Deep Ranking)
    Tab 2: Resume Screening (Original MLP)
    Tab 3: Manual Simulation (Original sliders + batch leaderboard)
"""

import streamlit as st
import numpy as np
import pandas as pd
import os
import sys
import json

# Original model imports
from model_base import ApplicantScreeningNN, min_max_normalize
from resume_parser import parse_resume

st.set_page_config(
    page_title="AI Recruitment Simulator - Neural Matching", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Model Loading
# ============================================================
@st.cache_resource
def load_original_model():
    """Load the original NumPy MLP model."""
    model_path = 'model_weights.npz'
    if not os.path.exists(model_path):
        return None
    nn = ApplicantScreeningNN()
    nn.load_model(model_path)
    return nn

@st.cache_resource
def load_neural_pipeline():
    """Load all 3 custom neural network models."""
    try:
        from inference_pipeline import CandidateMatchingPipeline
        pipeline = CandidateMatchingPipeline()
        if pipeline.load_models():
            return pipeline
        return None
    except Exception as e:
        st.warning(f"Could not load neural matching pipeline: {e}")
        return None


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


# ============================================================
# Main Application
# ============================================================
def main():
    # Sidebar
    st.sidebar.title("AI Recruitment System")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Custom Neural Networks")
    st.sidebar.markdown("""
    **1. Bi-LSTM + CRF**  
    Named Entity Recognition
    
    **2. Siamese Network**  
    Resume-JD Semantic Matching
    
    **3. Deep Ranking MLP**  
    Final Match Score
    """)
    st.sidebar.markdown("---")
    st.sidebar.markdown("Built with PyTorch + NumPy from scratch")
    
    # Title
    st.title("AI Recruitment Simulation System")
    st.caption("Custom Neural Networks for Intelligent Candidate Matching")
    
    # Load models
    nn = load_original_model()
    pipeline = load_neural_pipeline()
    
    # Create tabs
    if pipeline:
        tab1, tab2, tab3 = st.tabs([
            "Intelligent Matching (3 Networks)", 
            "Resume Screening (PDF)", 
            "Manual Simulation"
        ])
    else:
        tab2, tab3 = st.tabs([
            "Resume Screening (PDF)", 
            "Manual Simulation"
        ])
        tab1 = None
        st.info("Neural matching models not found. Run `python train_all_models.py` to enable Tab 1.")
    
    # ================================================================
    # TAB 1: Intelligent Candidate Matching (3 Neural Networks)
    # ================================================================
    if tab1 and pipeline:
        with tab1:
            st.header("Intelligent Candidate Matching")
            st.markdown("Uses **3 Custom Neural Networks** working together to match candidates to job descriptions.")
            
            # Two columns: JD input + Resume input
            col_jd, col_resume = st.columns(2)
            
            with col_jd:
                st.subheader("Job Description")
                jd_text = st.text_area(
                    "Enter the Job Description",
                    height=200,
                    placeholder="e.g., Looking for a Senior Data Scientist with expertise in Python, TensorFlow, and machine learning. Minimum 4 years of experience. Master's degree preferred.",
                    key="jd_input"
                )
            
            with col_resume:
                st.subheader("Resume Input")
                input_mode = st.radio("Input method:", ["Upload PDF", "Paste Text"], horizontal=True, key="input_mode")
                
                resume_text = ""
                if input_mode == "Upload PDF":
                    uploaded_file = st.file_uploader("Upload Resume PDF", type="pdf", key="pdf_neural")
                    if uploaded_file is not None:
                        try:
                            from resume_parser import extract_text_from_pdf
                            resume_text = extract_text_from_pdf(uploaded_file)
                            st.success(f"PDF parsed ({len(resume_text.split())} words extracted)")
                        except Exception as e:
                            st.error(f"Error parsing PDF: {e}")
                else:
                    resume_text = st.text_area(
                        "Paste resume text",
                        height=150,
                        placeholder="e.g., Experienced data scientist with 5 years in Python, TensorFlow, machine learning. Worked at Google. Master's in Computer Science from Stanford.",
                        key="resume_text_input"
                    )
            
            st.markdown("---")
            
            # Evaluate button
            if st.button("Evaluate Candidate", type="primary", use_container_width=True, key="eval_btn"):
                if not jd_text.strip():
                    st.warning("Please enter a Job Description.")
                elif not resume_text.strip():
                    st.warning("Please provide a resume (upload PDF or paste text).")
                else:
                    with st.spinner("Running 3-stage neural network pipeline..."):
                        result = pipeline.evaluate_candidate(resume_text, jd_text)
                    
                    # ---- Results Display ----
                    st.markdown("---")
                    
                    # Decision Banner
                    if result["decision"] == "SHORTLISTED":
                        st.success(f"**{result['decision']}** | Match Score: **{result['match_percentage']}%** | Confidence: {result['confidence']:.2f}")
                    else:
                        st.error(f"**{result['decision']}** | Match Score: **{result['match_percentage']}%** | Confidence: {result['confidence']:.2f}")
                    
                    # Three columns for the 3 network outputs
                    st.markdown("### Neural Network Outputs")
                    net1, net2, net3 = st.columns(3)
                    
                    with net1:
                        st.markdown("#### 1. Bi-LSTM + CRF (NER)")
                        st.caption("Extracted Entities from Resume")
                        entities = result["entities"]
                        
                        if entities.get("skills"):
                            st.markdown(f"**Skills:** {', '.join(entities['skills'])}")
                        else:
                            st.markdown("**Skills:** _None detected_")
                        
                        if entities.get("organizations"):
                            st.markdown(f"**Organizations:** {', '.join(entities['organizations'])}")
                        else:
                            st.markdown("**Organizations:** _None detected_")
                        
                        if entities.get("education"):
                            st.markdown(f"**Education:** {', '.join(entities['education'])}")
                        else:
                            st.markdown("**Education:** _None detected_")
                        
                        if entities.get("experience"):
                            st.markdown(f"**Experience:** {', '.join(entities['experience'])}")
                        else:
                            st.markdown("**Experience:** _None detected_")
                        
                        if entities.get("dates"):
                            st.markdown(f"**Dates:** {', '.join(entities['dates'])}")
                        else:
                            st.markdown("**Dates:** _None detected_")
                    
                    with net2:
                        st.markdown("#### 2. Siamese Network")
                        st.caption("Resume-JD Semantic Similarity")
                        
                        sim_score = result["siamese_similarity"]
                        st.metric("Semantic Similarity", f"{sim_score:.4f}")
                        st.progress(min(sim_score, 1.0))
                        
                        if sim_score > 0.7:
                            st.markdown("**Strong semantic match** between resume and job description.")
                        elif sim_score > 0.5:
                            st.markdown("**Moderate semantic match.** Some overlap in skills/domain.")
                        else:
                            st.markdown("**Weak semantic match.** Resume may not align with this role.")
                    
                    with net3:
                        st.markdown("#### 3. Deep Ranking MLP")
                        st.caption("Final Match Scoring")
                        
                        fv = result["feature_vector"]
                        st.metric("Final Match Score", f"{result['match_percentage']}%")
                        st.progress(min(result['match_score'], 1.0))
                        
                        st.markdown("**Input Feature Vector:**")
                        fv_df = pd.DataFrame({
                            "Feature": ["Siamese Similarity", "Skill Count", "Experience (Yrs)", 
                                       "Education Level", "Org Count", "Project Count"],
                            "Value": [fv["siamese_similarity"], fv["skill_count"], fv["experience_years"],
                                     fv["education_level"], fv["org_count"], fv["project_count"]]
                        })
                        st.dataframe(fv_df, hide_index=True, use_container_width=True)
            
            # ---- Batch Comparison ----
            st.markdown("---")
            st.subheader("Batch Candidate Comparison")
            st.caption("Compare multiple candidates against the same job description")
            
            num_candidates = st.number_input("Number of candidates to compare", min_value=2, max_value=10, value=3, key="num_cands")
            
            candidates = []
            cols = st.columns(min(num_candidates, 3))
            for i in range(num_candidates):
                with cols[i % 3]:
                    name = st.text_input(f"Candidate {i+1} Name", value=f"Candidate {i+1}", key=f"cand_name_{i}")
                    text = st.text_area(f"Resume Text", height=100, key=f"cand_text_{i}", 
                                       placeholder=f"Paste resume text for candidate {i+1}...")
                    if text.strip():
                        candidates.append({"name": name, "resume_text": text})
            
            if st.button("Rank All Candidates", key="rank_btn") and jd_text.strip() and len(candidates) >= 2:
                with st.spinner("Evaluating and ranking candidates..."):
                    results = pipeline.rank_multiple_candidates(candidates, jd_text)
                
                st.markdown("### Ranked Leaderboard")
                leaderboard_data = []
                for r in results:
                    leaderboard_data.append({
                        "Rank": r["rank"],
                        "Name": r["name"],
                        "Match Score (%)": r["match_percentage"],
                        "Similarity": r["siamese_similarity"],
                        "Skills Found": len(r["entities"].get("skills", [])),
                        "Decision": r["decision"],
                    })
                
                df_lb = pd.DataFrame(leaderboard_data)
                st.dataframe(df_lb, hide_index=True, use_container_width=True)
    
    # ================================================================
    # TAB 2: Original Resume Screening
    # ================================================================
    with tab2:
        st.header("Upload a Resume for AI Screening")
        
        if nn is None:
            st.error("Model file model_weights.npz not found! Please run train.py first.")
            return
        
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_original")
        
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
                st.success(f"**{results['Decision']}** | Fit Score: {results['Fit Score']:.2f}%")
            else:
                st.error(f"**{results['Decision']}** | Fit Score: {results['Fit Score']:.2f}%")
                
    # ================================================================
    # TAB 3: Manual Simulation
    # ================================================================
    with tab3:
        st.header("Manual Candidate Simulation")
        
        if nn is None:
            st.error("Model file model_weights.npz not found!")
            return
        
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
