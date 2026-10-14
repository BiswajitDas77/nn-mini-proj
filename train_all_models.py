"""
Unified Training Script for All 3 Custom Neural Networks (OPTIMIZED)
=====================================================================
Trains the complete candidate matching pipeline with optimized parameters
for fast CPU training while maintaining model quality.

Usage: python train_all_models.py
"""

import time
import os
import sys

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    start_time = time.time()
    
    print("\n" + "#" * 60)
    print("#  CUSTOM NEURAL NETWORKS FOR INTELLIGENT CANDIDATE MATCHING")
    print("#  Training Pipeline (Optimized for CPU)")
    print("#" * 60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # ============================
    # Phase 1: Bi-LSTM + CRF NER
    # ============================
    print("\n\n" + "=" * 60)
    print("  PHASE 1: Bi-LSTM + CRF (Named Entity Recognition)")
    print("=" * 60)
    phase1_start = time.time()
    
    from bilstm_crf_model import train_ner_model, extract_entities
    ner_model, word_to_ix = train_ner_model(
        num_samples=500,   # Reduced from 2000 for CPU speed
        epochs=10,         # Reduced from 25 - sufficient for convergence
        lr=0.015,
        save_path=os.path.join(base_dir, "ner_model.pth")
    )
    
    # Quick verification
    test_text = "experienced in python and machine learning worked at google for 3 years bachelor in computer science from mit"
    entities = extract_entities(test_text, ner_model, word_to_ix)
    print(f"\n  Verification test:")
    print(f"  Input:  '{test_text}'")
    print(f"  Output: {entities}")
    
    phase1_time = time.time() - phase1_start
    print(f"\n  Phase 1 completed in {phase1_time:.1f}s")
    
    # ============================
    # Phase 2: Siamese Network
    # ============================
    print("\n\n" + "=" * 60)
    print("  PHASE 2: Siamese Neural Network (Resume-JD Matching)")
    print("=" * 60)
    phase2_start = time.time()
    
    from siamese_model import train_siamese_model, compute_similarity
    siamese_model, vectorizer = train_siamese_model(
        num_pairs=2000,    # Reduced from 3000
        epochs=20,         # Reduced from 30
        lr=0.001,
        save_path=os.path.join(base_dir, "siamese_model.pth")
    )
    
    # Quick verification
    resume = "python developer experienced in machine learning tensorflow data science 5 years"
    jd_match = "looking for data scientist with python tensorflow machine learning experience"
    jd_nomatch = "hiring frontend developer react javascript angular html css"
    
    sim_match = compute_similarity(resume, jd_match, siamese_model, vectorizer)
    sim_nomatch = compute_similarity(resume, jd_nomatch, siamese_model, vectorizer)
    
    print(f"\n  Verification test:")
    print(f"  Resume:  '{resume[:60]}...'")
    print(f"  Match JD similarity:     {sim_match:.4f}")
    print(f"  Non-match JD similarity: {sim_nomatch:.4f}")
    print(f"  Correct ordering: {'YES' if sim_match > sim_nomatch else 'NO'}")
    
    phase2_time = time.time() - phase2_start
    print(f"\n  Phase 2 completed in {phase2_time:.1f}s")
    
    # ============================
    # Phase 3: Deep Ranking MLP
    # ============================
    print("\n\n" + "=" * 60)
    print("  PHASE 3: Deep Ranking Network MLP (Final Match Scoring)")
    print("=" * 60)
    phase3_start = time.time()
    
    from deep_ranking_model import train_ranking_model, rank_candidates
    import numpy as np
    
    ranking_model, normalizer = train_ranking_model(
        num_samples=3000,  # Reduced from 4000
        epochs=30,         # Reduced from 40
        lr=0.001,
        save_path=os.path.join(base_dir, "ranking_model.pth")
    )
    
    # Quick verification
    test_candidates = [
        [0.9, 12, 7.0, 3, 4, 6],  # Strong
        [0.7, 8, 4.5, 2, 2, 3],   # Moderate
        [0.5, 5, 2.0, 2, 1, 2],   # Below average
        [0.3, 3, 1.0, 1, 1, 1],   # Weak
    ]
    
    ranked = rank_candidates(np.array(test_candidates), ranking_model, normalizer)
    print(f"\n  Verification test (Candidate Ranking):")
    for rank, (idx, score) in enumerate(ranked, 1):
        labels = ["Strong", "Moderate", "Below Avg", "Weak"]
        print(f"    #{rank}: {labels[idx]:>10s} candidate -> Score: {score:.4f}")
    
    phase3_time = time.time() - phase3_start
    print(f"\n  Phase 3 completed in {phase3_time:.1f}s")
    
    # ============================
    # Summary
    # ============================
    total_time = time.time() - start_time
    
    print("\n\n" + "#" * 60)
    print("#  TRAINING COMPLETE -- ALL 3 MODELS SAVED")
    print("#" * 60)
    print(f"\n  Model Files:")
    print(f"    1. ner_model.pth      (Bi-LSTM + CRF)      [{phase1_time:.1f}s]")
    print(f"    2. siamese_model.pth  (Siamese Network)     [{phase2_time:.1f}s]")
    print(f"    3. ranking_model.pth  (Deep Ranking MLP)    [{phase3_time:.1f}s]")
    print(f"\n  Total training time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
