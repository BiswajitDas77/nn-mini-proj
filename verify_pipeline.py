import os
import csv
import numpy as np
from resume_parser import parse_resume

def print_header(text):
    print(f"\n{'='*60}\n[ {text} ]\n{'-'*60}")
    
def print_status(status, details=""):
    color_start = "\033[92m" if status == "PASS" else "\033[91m"
    color_end = "\033[0m"
    print(f"[{color_start}{status}{color_end}] {details}")

def generate_sample_pdfs():
    from fpdf import FPDF
    os.makedirs('sample_resumes', exist_ok=True)
    
    # Strong Candidate
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    strong_text = "Experienced software engineer with 9 years of experience. Highly skilled in Python, Java, C++, Docker, AWS, Node, React, SQL, and Machine Learning. Managed a team, showing excellent leadership, teamwork, communication, problem solving, time management, and agile skills. Led multiple projects, developed scalable applications, built strong data pipelines, created robust APIs, designed architectures, and developed interfaces."
    pdf.multi_cell(0, 10, strong_text)
    pdf.output("sample_resumes/strong_candidate.pdf")
    
    # Weak Candidate
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    weak_text = "Junior developer looking for opportunities. 0.5 years of experience. Familiar with HTML and some Python. Basic communication skills."
    pdf.multi_cell(0, 10, weak_text)
    pdf.output("sample_resumes/weak_candidate.pdf")

def phase1_dataset():
    print_header("PHASE 1: DATASET INTEGRITY CHECK")
    all_pass = True
    
    for filename, expected_min_rows in [("train_candidates.csv", 400), ("test_candidates.csv", 100)]:
        if not os.path.exists(filename):
            print_status("FAIL", f"{filename} missing.")
            all_pass = False
            continue
            
        with open(filename, 'r') as f:
            reader = list(csv.reader(f))
            header = reader[0]
            data = reader[1:]
            
        if len(data) >= expected_min_rows:
            print_status("PASS", f"{filename} row count = {len(data)} (>= {expected_min_rows})")
        else:
            print_status("FAIL", f"{filename} row count = {len(data)} (< {expected_min_rows})")
            all_pass = False
            
        if len(header) == 5 and "shortlisted" in header:
            print_status("PASS", f"{filename} has 5 correct columns.")
        else:
            print_status("FAIL", f"{filename} columns incorrect: {header}")
            all_pass = False
            
        # Check numerical ranges and labels
        valid = True
        for row in data:
            try:
                exp = float(row[0])
                tech = float(row[1])
                soft = float(row[2])
                proj = int(row[3])
                label = int(row[4])
                if not (0 <= exp <= 15 and 0 <= tech <= 100 and 0 <= soft <= 100 and 0 <= proj <= 10 and label in [0, 1]):
                    valid = False
            except:
                valid = False
                
        if valid:
            print_status("PASS", f"{filename} features in expected range and labels strictly binary.")
        else:
            print_status("FAIL", f"{filename} data validation failed.")
            all_pass = False
            
    return all_pass

def phase2_model_weights():
    print_header("PHASE 2: MODEL WEIGHT ARTIFACTS AUDIT")
    model_path = 'model_weights.npz'
    if not os.path.exists(model_path):
        print_status("FAIL", f"{model_path} missing.")
        return False, None
        
    data = np.load(model_path)
    required_keys = ['W1', 'b1', 'W2', 'b2', 'min_vals', 'max_vals']
    all_pass = True
    
    for key in required_keys:
        if key not in data:
            print_status("FAIL", f"Missing matrix {key}")
            all_pass = False
            continue
            
        mat = data[key]
        
        # Check NaN / Inf
        if np.isnan(mat).any() or np.isinf(mat).any():
            print_status("FAIL", f"Matrix {key} contains NaN or Inf.")
            all_pass = False
        else:
            print_status("PASS", f"Matrix {key} shape: {mat.shape} (No NaN/Inf)")
            
    # Verify exact shapes
    try:
        assert data['W1'].shape == (4, 6)
        assert data['b1'].shape == (1, 6)
        assert data['W2'].shape == (6, 1)
        assert data['b2'].shape == (1, 1)
        assert data['min_vals'].shape in [(4,), (1, 4)]
        assert data['max_vals'].shape in [(4,), (1, 4)]
        print_status("PASS", "All weight matrices match expected architecture (4 -> 6 -> 1).")
    except AssertionError:
        print_status("FAIL", "Matrix shapes mismatch!")
        all_pass = False
        
    return all_pass, data

def manual_forward(X, data):
    W1, b1, W2, b2 = data['W1'], data['b1'], data['W2'], data['b2']
    min_vals, max_vals = data['min_vals'], data['max_vals']
    
    # Min-max scale
    X_norm = (X - min_vals) / (max_vals - min_vals + 1e-8)
    
    # Fwd Pass
    Z1 = np.dot(X_norm, W1) + b1
    A1 = np.maximum(0, Z1) # ReLU
    Z2 = np.dot(A1, W2) + b2
    
    # Sigmoid
    Z2_clipped = np.clip(Z2, -500, 500)
    A2 = 1.0 / (1.0 + np.exp(-Z2_clipped))
    
    return A2[0][0]

def phase3_manual_pass(data):
    print_header("PHASE 3: MANUAL NUMPY FORWARD PASS & MATH SANITY CHECK")
    if data is None:
        print_status("FAIL", "Missing model data, skipping Phase 3.")
        return False
        
    X_A = np.array([[6.0, 90.0, 85.0, 5]])
    X_B = np.array([[0.5, 30.0, 40.0, 0]])
    
    score_A = manual_forward(X_A, data)
    score_B = manual_forward(X_B, data)
    
    print(f"Profile A (Expected Hire) Score:   {score_A:.4f}")
    print(f"Profile B (Expected Reject) Score: {score_B:.4f}")
    
    if 0.0 < score_A < 1.0 and 0.0 < score_B < 1.0:
        print_status("PASS", "Probabilities fall strictly within (0.0, 1.0).")
    else:
        print_status("FAIL", "Probabilities out of bounds!")
        return False
        
    if score_A > score_B:
        print_status("PASS", "Score(Profile A) > Score(Profile B) (Sanity Check passed).")
        return True
    else:
        print_status("FAIL", "Math logic incorrect! Profile B scored higher than Profile A.")
        return False

def phase4_resume_parser():
    print_header("PHASE 4: RESUME PDF PARSER AUDIT")
    if not (os.path.exists("sample_resumes/strong_candidate.pdf") and os.path.exists("sample_resumes/weak_candidate.pdf")):
        try:
            generate_sample_pdfs()
            print_status("PASS", "Generated sample_resumes/strong_candidate.pdf and weak_candidate.pdf")
        except Exception as e:
            print_status("FAIL", f"Failed to generate PDFs: {e}")
            return False, None, None
            
    strong_features = parse_resume("sample_resumes/strong_candidate.pdf")
    weak_features = parse_resume("sample_resumes/weak_candidate.pdf")
    
    print(f"Strong Candidate Parsed: {strong_features}")
    print(f"Weak Candidate Parsed:   {weak_features}")
    
    def validate(f):
        return isinstance(f, dict) and len(f) == 4 and all(v >= 0 for v in f.values())
        
    if validate(strong_features) and validate(weak_features):
        print_status("PASS", "Parsed extracted valid 4-element numeric vector.")
        return True, strong_features, weak_features
    else:
        print_status("FAIL", "Invalid parser output format or negative values.")
        return False, None, None

def phase5_end_to_end(strong_f, weak_f, data):
    print_header("PHASE 5: END-TO-END INFERENCE & DISCRIMINATION CHECK")
    if data is None or strong_f is None or weak_f is None:
        print_status("FAIL", "Missing dependencies from prior phases.")
        return False
        
    X_strong = np.array([[strong_f['Experience'], strong_f['Tech_Score'], strong_f['Soft_Score'], strong_f['Projects']]])
    X_weak = np.array([[weak_f['Experience'], weak_f['Tech_Score'], weak_f['Soft_Score'], weak_f['Projects']]])
    
    score_strong = manual_forward(X_strong, data) * 100
    score_weak = manual_forward(X_weak, data) * 100
    
    print(f"Strong Candidate Final Fit Score: {score_strong:.2f}%")
    print(f"Weak Candidate Final Fit Score:   {score_weak:.2f}%")
    
    all_pass = True
    if score_strong >= 60.0:
        print_status("PASS", f"Strong candidate safely shortlisted (>= 60%).")
    else:
        print_status("FAIL", f"Strong candidate incorrectly rejected.")
        all_pass = False
        
    if score_weak < 50.0:
        print_status("PASS", f"Weak candidate correctly rejected (< 50%).")
    else:
        print_status("FAIL", f"Weak candidate incorrectly shortlisted.")
        all_pass = False
        
    return all_pass

def main():
    p1 = phase1_dataset()
    p2, model_data = phase2_model_weights()
    p3 = phase3_manual_pass(model_data)
    p4, strong_f, weak_f = phase4_resume_parser()
    p5 = phase5_end_to_end(strong_f, weak_f, model_data)
    
    print("\n" + "="*70)
    if all([p1, p2, p3, p4, p5]):
        print("\033[92mOVERALL SYSTEM STATUS: ALL PHASES VERIFIED (READY FOR DEMO / VIVA)\033[0m")
    else:
        print("\033[91mOVERALL SYSTEM STATUS: FAILURES DETECTED IN AUDIT\033[0m")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
