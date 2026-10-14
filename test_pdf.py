"""
Test PDF Resume parsing and Intelligent Neural Matching Pipeline
"""
import os
import sys

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from resume_parser import extract_text_from_pdf
from inference_pipeline import CandidateMatchingPipeline

def main():
    pipeline = CandidateMatchingPipeline()
    if not pipeline.load_models():
        print("Failed to load models!")
        return

    jd = """
    We are hiring a Senior Software Engineer with strong expertise in Python, Docker, AWS, 
    and Machine Learning. Must have at least 5 years of professional experience and proven leadership.
    """

    for pdf_name in ["strong_candidate.pdf", "weak_candidate.pdf"]:
        pdf_path = os.path.join("sample_resumes", pdf_name)
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                text = extract_text_from_pdf(f)
            
            result = pipeline.evaluate_candidate(text, jd)
            print("=" * 60)
            print(f"File: {pdf_name}")
            print(f"Decision: {result['decision']}")
            print(f"Match Score: {result['match_percentage']}%")
            print(f"Siamese Similarity: {result['siamese_similarity']:.4f}")
            print(f"Extracted Skills: {result['entities'].get('skills', [])}")
            print(f"Extracted Experience: {result['entities'].get('experience', [])}")
            print("=" * 60)

if __name__ == "__main__":
    main()
