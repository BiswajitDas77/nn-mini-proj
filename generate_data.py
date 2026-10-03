import numpy as np
import csv

def generate_dataset(num_samples, seed, filename):
    np.random.seed(seed)
    
    X = np.zeros((num_samples, 4))
    y = np.zeros((num_samples, 1), dtype=int)
    
    for i in range(num_samples):
        # 1. Generate Features
        exp = np.random.uniform(0.0, 10.0)
        tech = np.random.uniform(20.0, 100.0)
        soft = np.random.uniform(25.0, 100.0)
        proj = np.random.randint(0, 7)
        
        X[i] = [exp, tech, soft, proj]
        
        # 2. Base Composite Formula
        # Normalized Experience = (exp / 10.0) * 100.0
        # Normalized Projects = (proj / 6.0) * 100.0
        base_score = (tech * 0.40) + (soft * 0.25) + ((exp / 10.0 * 100.0) * 0.20) + ((proj / 6.0 * 100.0) * 0.15)
        
        # 3. Inject Gaussian Noise (std_dev = 5.0 for balanced subjectivity)
        noise = np.random.normal(0, 5.0)
        final_score = base_score + noise
        
        # 4. Label Assignment Logic
        if final_score >= 67.0 and tech >= 50.0:
            y[i] = 1
        else:
            y[i] = 0
            
        # Inject realistic edge cases: 3% random label flip
        if np.random.rand() < 0.03:
            y[i] = 1 - y[i]
            
    # Export to CSV
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['years_experience', 'technical_score', 'soft_skills_score', 'projects_completed', 'shortlisted'])
        for i in range(num_samples):
            writer.writerow([
                round(X[i][0], 2), 
                round(X[i][1], 2), 
                round(X[i][2], 2), 
                int(X[i][3]), 
                int(y[i][0])
            ])
            
    print(f"Successfully generated {filename} with {num_samples} rows (Seed: {seed}).")

if __name__ == '__main__':
    # Generate Training Data
    generate_dataset(num_samples=450, seed=42, filename='train_candidates.csv')
    
    # Generate Testing Data
    generate_dataset(num_samples=100, seed=101, filename='test_candidates.csv')
