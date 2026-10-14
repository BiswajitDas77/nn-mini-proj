"""
Unified Inference Pipeline — Intelligent Candidate Matching
============================================================
End-to-end pipeline that connects all 3 custom neural networks:

    Resume PDF + Job Description Text
        ↓
    [1] BiLSTM-CRF NER → Extract entities (skills, orgs, education, dates, experience)
        ↓
    [2] Siamese Network → Compute resume-JD semantic similarity
        ↓
    [3] Deep Ranking MLP → Produce final match score using NER features + similarity
        ↓
    Final Output: Match Score + Ranked Position + Extracted Entities

This module is imported by the Streamlit app for live inference.
"""

import os
import sys
import re
import numpy as np
import torch

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Import model modules
from bilstm_crf_model import load_ner_model, extract_entities, BiLSTM_CRF
from siamese_model import load_siamese_model, compute_similarity, TextVectorizer, SiameseNetwork
from deep_ranking_model import load_ranking_model, rank_candidates, DeepRankingMLP, FeatureNormalizer


# ============================================================
# Education Level Encoder
# ============================================================
EDUCATION_HIERARCHY = {
    "phd": 4, "doctorate": 4,
    "master": 3, "mtech": 3, "msc": 3, "mca": 3, "mba": 3, "ms": 3, "postgraduate": 3,
    "bachelor": 2, "btech": 2, "bsc": 2, "bca": 2, "be": 2, "ba": 2, "graduate": 2, "degree": 2,
    "diploma": 1, "certification": 1, "certificate": 1,
}

def encode_education_level(education_entities):
    """
    Convert education entities to a numerical level (0-4).
    Takes the highest education level found.
    """
    if not education_entities:
        return 0
    
    max_level = 0
    for entity in education_entities:
        for word in entity.lower().split():
            if word in EDUCATION_HIERARCHY:
                max_level = max(max_level, EDUCATION_HIERARCHY[word])
    
    return max_level


def parse_experience_years(experience_entities):
    """
    Extract total years of experience from NER experience entities.
    E.g., ["3 years", "6 months"] → 3.5
    """
    if not experience_entities:
        return 0.0
    
    total_years = 0.0
    for entity in experience_entities:
        # Look for year patterns
        year_match = re.findall(r'(\d+(?:\.\d+)?)\s*(?:years?|yrs?)', entity)
        month_match = re.findall(r'(\d+)\s*(?:months?|mos?)', entity)
        
        total_years += sum(float(y) for y in year_match)
        total_years += sum(float(m) / 12.0 for m in month_match)
    
    return min(total_years, 10.0)  # Cap at 10


# ============================================================
# CandidateMatchingPipeline
# ============================================================
class CandidateMatchingPipeline:
    """
    Complete end-to-end pipeline for intelligent candidate matching.
    
    Usage:
        pipeline = CandidateMatchingPipeline()
        pipeline.load_models()
        result = pipeline.evaluate_candidate(resume_text, job_description)
    """
    
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = model_dir
        
        self.ner_model = None
        self.word_to_ix = None
        self.siamese_model = None
        self.vectorizer = None
        self.ranking_model = None
        self.normalizer = None
        self.loaded = False
    
    def load_models(self):
        """Load all 3 trained models from disk."""
        print("Loading models...")
        
        # 1. BiLSTM-CRF NER
        ner_path = os.path.join(self.model_dir, "ner_model.pth")
        if os.path.exists(ner_path):
            self.ner_model, self.word_to_ix = load_ner_model(ner_path)
            print("  [OK] Bi-LSTM + CRF NER model loaded")
        else:
            print(f"  [X] NER model not found at {ner_path}")
            return False
        
        # 2. Siamese Network
        siamese_path = os.path.join(self.model_dir, "siamese_model.pth")
        if os.path.exists(siamese_path):
            self.siamese_model, self.vectorizer = load_siamese_model(siamese_path)
            print("  [OK] Siamese Network model loaded")
        else:
            print(f"  [X] Siamese model not found at {siamese_path}")
            return False
        
        # 3. Deep Ranking MLP
        ranking_path = os.path.join(self.model_dir, "ranking_model.pth")
        if os.path.exists(ranking_path):
            self.ranking_model, self.normalizer = load_ranking_model(ranking_path)
            print("  [OK] Deep Ranking MLP model loaded")
        else:
            print(f"  [X] Ranking model not found at {ranking_path}")
            return False
        
        self.loaded = True
        print("All models loaded successfully!\n")
        return True
    
    def extract_resume_entities(self, resume_text):
        """
        Step 1: Use BiLSTM-CRF to extract named entities from resume.
        
        Returns dict with keys: skills, organizations, education, dates, experience
        """
        if not self.ner_model:
            return {"skills": [], "organizations": [], "education": [], "dates": [], "experience": []}
        
        return extract_entities(resume_text, self.ner_model, self.word_to_ix)
    
    def compute_resume_jd_similarity(self, resume_text, jd_text):
        """
        Step 2: Use Siamese Network to compute semantic similarity.
        
        Returns float between 0.0 and 1.0
        """
        if not self.siamese_model:
            return 0.5
        
        return compute_similarity(resume_text, jd_text, self.siamese_model, self.vectorizer)
    
    def compute_final_score(self, siamese_score, entities):
        """
        Step 3: Use Deep Ranking MLP to compute final match score.
        
        Combines Siamese similarity with NER-extracted features.
        
        Returns float between 0.0 and 1.0
        """
        if not self.ranking_model:
            return siamese_score
        
        # Build feature vector
        skill_count = len(entities.get("skills", []))
        exp_years = parse_experience_years(entities.get("experience", []))
        edu_level = encode_education_level(entities.get("education", []))
        org_count = len(entities.get("organizations", []))
        
        # Estimate project count from skill diversity
        proj_count = min(skill_count // 2, 6)
        
        features = np.array([[
            siamese_score,
            skill_count,
            exp_years,
            edu_level,
            org_count,
            proj_count
        ]], dtype=np.float32)
        
        features_norm = self.normalizer.transform(features)
        
        with torch.no_grad():
            X = torch.FloatTensor(features_norm)
            score = self.ranking_model(X).squeeze().item()
        
        return score
    
    def evaluate_candidate(self, resume_text, job_description):
        """
        Full end-to-end evaluation of a candidate against a job description.
        
        Args:
            resume_text: Raw resume text (from PDF or direct input)
            job_description: Job description text
        
        Returns:
            dict with:
                - entities: Extracted named entities from resume
                - siamese_similarity: Semantic similarity score
                - match_score: Final calibrated match score
                - decision: "SHORTLISTED" or "REJECTED"
                - feature_vector: The 6-dim input to ranking MLP
                - confidence: How confident the decision is
        """
        if not self.loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Step 1: Extract entities via BiLSTM-CRF
        entities = self.extract_resume_entities(resume_text)
        
        # Step 2: Compute similarity via Siamese Network
        siamese_score = self.compute_resume_jd_similarity(resume_text, job_description)
        
        # Step 3: Compute final score via Deep Ranking MLP
        skill_count = len(entities.get("skills", []))
        exp_years = parse_experience_years(entities.get("experience", []))
        edu_level = encode_education_level(entities.get("education", []))
        org_count = len(entities.get("organizations", []))
        proj_count = min(skill_count // 2, 6)
        
        feature_vector = [siamese_score, skill_count, exp_years, edu_level, org_count, proj_count]
        
        match_score = self.compute_final_score(siamese_score, entities)
        
        # Decision threshold
        decision = "SHORTLISTED" if match_score >= 0.5 else "REJECTED"
        confidence = abs(match_score - 0.5) * 2  # 0 = uncertain, 1 = very confident
        
        return {
            "entities": entities,
            "siamese_similarity": round(siamese_score, 4),
            "match_score": round(match_score, 4),
            "match_percentage": round(match_score * 100, 1),
            "decision": decision,
            "confidence": round(confidence, 4),
            "feature_vector": {
                "siamese_similarity": round(siamese_score, 4),
                "skill_count": skill_count,
                "experience_years": round(exp_years, 2),
                "education_level": edu_level,
                "org_count": org_count,
                "project_count": proj_count,
            }
        }
    
    def rank_multiple_candidates(self, candidates, job_description):
        """
        Evaluate and rank multiple candidates against a single job description.
        
        Args:
            candidates: list of dicts with "name" and "resume_text" keys
            job_description: Job description text
        
        Returns:
            Sorted list of evaluation results with candidate names and ranks
        """
        results = []
        
        for candidate in candidates:
            result = self.evaluate_candidate(candidate["resume_text"], job_description)
            result["name"] = candidate.get("name", "Unknown")
            results.append(result)
        
        # Sort by match score descending
        results.sort(key=lambda x: x["match_score"], reverse=True)
        
        # Add rank
        for i, result in enumerate(results):
            result["rank"] = i + 1
        
        return results


# ============================================================
# Main (standalone test)
# ============================================================
if __name__ == "__main__":
    pipeline = CandidateMatchingPipeline()
    
    if not pipeline.load_models():
        print("ERROR: Models not found. Run train_all_models.py first.")
        exit(1)
    
    # Test with sample data
    job_description = """
    Looking for a Senior Data Scientist with expertise in Python, TensorFlow, 
    and machine learning. Minimum 4 years of experience required. 
    Master's degree in Computer Science or Data Science preferred.
    Experience with data pipeline and analytics is a plus.
    """
    
    candidates = [
        {
            "name": "Alice (Strong Match)",
            "resume_text": "experienced data scientist with 5 years of experience proficient in python tensorflow machine learning deep learning worked at google master degree in computer science from stanford built multiple data pipeline projects"
        },
        {
            "name": "Bob (Moderate Match)",
            "resume_text": "software engineer with 3 years experience in python django sql worked at infosys bachelor in computer science from iit built web applications"
        },
        {
            "name": "Charlie (Weak Match)",
            "resume_text": "frontend developer skilled in javascript react angular html css 2 years experience worked at startup bachelor degree"
        },
    ]
    
    print("\n" + "=" * 60)
    print("  END-TO-END PIPELINE TEST")
    print("=" * 60)
    print(f"\nJob Description: {job_description.strip()[:100]}...")
    
    results = pipeline.rank_multiple_candidates(candidates, job_description)
    
    for result in results:
        print(f"\n{'-' * 50}")
        print(f"  Rank #{result['rank']}: {result['name']}")
        print(f"  Decision: {result['decision']}")
        print(f"  Match Score: {result['match_percentage']}%")
        print(f"  Siamese Similarity: {result['siamese_similarity']}")
        print(f"  Confidence: {result['confidence']}")
        print(f"  Entities: {result['entities']}")
        print(f"  Feature Vector: {result['feature_vector']}")
