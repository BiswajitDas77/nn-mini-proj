"""
Siamese Neural Network for Resume–Job Description Semantic Matching
===================================================================
Processes both the resume and job description using two identical branches
with shared weights. Learns semantic similarity using Contrastive Loss.

Architecture:
    Resume Text → TextVectorizer → [Shared MLP Branch] → Resume Embedding
    JD Text     → TextVectorizer → [Shared MLP Branch] → JD Embedding
    
    Contrastive Loss pulls matching pairs together and pushes
    non-matching pairs apart in the embedding space.
    
    Similarity = 1 - normalized_euclidean_distance(resume_emb, jd_emb)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import json
import os


# ============================================================
# Text Vectorizer (Keyword-Based Feature Extraction)
# ============================================================

# Comprehensive keyword vocabulary for recruitment domain
KEYWORD_VOCABULARY = [
    # Programming Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "ruby", "golang",
    "rust", "swift", "kotlin", "scala", "matlab", "r", "php", "perl",
    
    # Frameworks & Libraries
    "react", "angular", "vue", "django", "flask", "spring", "nodejs", "express",
    "tensorflow", "pytorch", "keras", "pandas", "numpy", "sklearn", "scipy",
    "bootstrap", "jquery", "nextjs", "fastapi", "graphql",
    
    # Cloud & DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "terraform",
    "ansible", "ci/cd", "microservices", "serverless", "lambda",
    
    # Databases
    "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    "cassandra", "dynamodb", "oracle", "firebase",
    
    # Data & AI
    "machine", "learning", "deep", "neural", "nlp", "computer", "vision",
    "data", "science", "analytics", "statistics", "bigdata", "spark",
    "hadoop", "kafka", "etl", "pipeline", "warehouse", "mining",
    "ai", "artificial", "intelligence", "reinforcement",
    
    # Soft Skills & General
    "leadership", "communication", "teamwork", "management", "agile",
    "scrum", "collaboration", "problem", "solving", "critical", "thinking",
    "presentation", "mentoring", "planning", "strategy",
    
    # Roles & Levels
    "engineer", "developer", "architect", "analyst", "scientist", "manager",
    "senior", "junior", "lead", "principal", "intern", "associate",
    "fullstack", "frontend", "backend", "devops", "cloud",
    
    # Education
    "bachelor", "master", "phd", "degree", "certification", "diploma",
    "university", "college", "graduate", "postgraduate",
    
    # Experience indicators  
    "years", "experience", "professional", "worked", "built", "developed",
    "designed", "implemented", "deployed", "optimized", "maintained",
    "created", "managed", "led", "delivered", "contributed",
]

VOCAB_SIZE = len(KEYWORD_VOCABULARY)


class TextVectorizer:
    """
    Converts raw text into a fixed-size numerical feature vector
    based on keyword frequency in the recruitment domain vocabulary.
    
    Each dimension corresponds to the normalized frequency of a 
    domain-specific keyword appearing in the text.
    """
    
    def __init__(self, vocabulary=KEYWORD_VOCABULARY):
        self.vocabulary = vocabulary
        self.word_to_idx = {w: i for i, w in enumerate(vocabulary)}
        self.vector_size = len(vocabulary)
    
    def vectorize(self, text):
        """
        Convert text to keyword frequency vector.
        
        Args:
            text: Raw text string
        Returns:
            numpy array of shape (vector_size,) with normalized keyword frequencies
        """
        words = text.lower().split()
        vector = np.zeros(self.vector_size, dtype=np.float32)
        
        word_count = len(words) if words else 1
        
        for word in words:
            # Clean punctuation
            word = word.strip(".,;:!?()[]{}\"'")
            if word in self.word_to_idx:
                vector[self.word_to_idx[word]] += 1.0
        
        # Normalize by document length (TF normalization)
        vector = vector / word_count
        
        return vector
    
    def vectorize_batch(self, texts):
        """Vectorize a list of texts."""
        return np.array([self.vectorize(t) for t in texts])


# ============================================================
# Siamese Neural Network
# ============================================================
class SiameseNetwork(nn.Module):
    """
    Siamese Neural Network with shared weights.
    
    Both branches use the SAME weights (shared_network) to project
    text feature vectors into a learned embedding space where semantic
    similarity is captured by Euclidean distance.
    
    Architecture per branch:
        Input(vocab_size) → Linear(256) → BatchNorm → ReLU → Dropout(0.3)
                          → Linear(128) → BatchNorm → ReLU → Dropout(0.2)
                          → Linear(64)  → L2 Normalize → Embedding(64)
    """
    
    def __init__(self, input_dim=VOCAB_SIZE, embedding_dim=64):
        super(SiameseNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # Shared network (identical weights for both branches)
        self.shared_network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, embedding_dim),
        )
    
    def forward_one(self, x):
        """Forward pass through one branch."""
        output = self.shared_network(x)
        # L2 normalize embeddings for stable distance computation
        output = F.normalize(output, p=2, dim=1)
        return output
    
    def forward(self, resume_input, jd_input):
        """
        Forward pass through both branches.
        
        Args:
            resume_input: (batch, input_dim) resume feature vectors
            jd_input: (batch, input_dim) job description feature vectors
        
        Returns:
            resume_embedding: (batch, embedding_dim) 
            jd_embedding: (batch, embedding_dim)
        """
        resume_embedding = self.forward_one(resume_input)
        jd_embedding = self.forward_one(jd_input)
        return resume_embedding, jd_embedding
    
    def get_similarity(self, resume_input, jd_input):
        """
        Compute similarity score between resume and JD (0 to 1).
        
        Uses: similarity = 1 - normalized_euclidean_distance
        """
        with torch.no_grad():
            emb1, emb2 = self.forward(resume_input, jd_input)
            distance = F.pairwise_distance(emb1, emb2)
            # Normalize distance to [0, 1] and invert so 1 = most similar
            similarity = 1.0 / (1.0 + distance)
        return similarity


# ============================================================
# Contrastive Loss
# ============================================================
class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for Siamese Networks.
    
    For matching pairs (label=1): minimizes the distance between embeddings.
    For non-matching pairs (label=0): pushes embeddings apart beyond a margin.
    
    L = (1/2N) Σ [ y · D² + (1-y) · max(0, margin - D)² ]
    
    where:
        y = 1 if pair is a match, 0 if non-match
        D = Euclidean distance between embeddings
        margin = minimum distance threshold for non-matching pairs
    """
    
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, embedding1, embedding2, label):
        """
        Args:
            embedding1: (batch, emb_dim) first branch output
            embedding2: (batch, emb_dim) second branch output
            label: (batch,) 1.0 for matching pairs, 0.0 for non-matching
        
        Returns:
            Scalar loss value
        """
        # Euclidean distance
        distance = F.pairwise_distance(embedding1, embedding2)
        
        # Contrastive loss
        loss_match = label * distance.pow(2)
        loss_nonmatch = (1 - label) * F.relu(self.margin - distance).pow(2)
        
        loss = 0.5 * (loss_match + loss_nonmatch)
        return loss.mean()


# ============================================================
# Synthetic Training Data Generator
# ============================================================

# Job description templates
JD_TEMPLATES = {
    "software_engineer": [
        "looking for software engineer with experience in {skills}. required {exp} years of experience. {edu} degree preferred.",
        "we need a developer proficient in {skills}. minimum {exp} years. {edu} in computer science required.",
        "hiring software engineer skilled in {skills} with {exp} years experience and {edu} degree.",
    ],
    "data_scientist": [
        "seeking data scientist with expertise in {skills}. {exp} years of experience needed. {edu} in data science or statistics.",
        "data scientist role requiring {skills}. minimum {exp} years. {edu} degree in relevant field.",
        "looking for data science professional with {skills} knowledge and {exp} years experience.",
    ],
    "devops_engineer": [
        "devops engineer needed with skills in {skills}. {exp} years of experience. {edu} in computer science.",
        "hiring devops engineer proficient in {skills} with {exp} years experience.",
        "seeking cloud devops engineer skilled in {skills}. requires {exp} years and {edu} degree.",
    ],
    "frontend_developer": [
        "frontend developer with {skills} expertise needed. {exp} years of experience. {edu} degree required.",
        "we are hiring a frontend developer skilled in {skills} with {exp} years experience.",
        "looking for frontend engineer with {skills}. minimum {exp} years. {edu} preferred.",
    ],
    "ml_engineer": [
        "machine learning engineer with {skills} experience. {exp} years required. {edu} in ai or ml.",
        "seeking ml engineer proficient in {skills}. minimum {exp} years experience. {edu} degree needed.",
        "hiring ml engineer with deep knowledge of {skills} and {exp} years experience.",
    ],
}

ROLE_SKILLS = {
    "software_engineer": [
        "python java spring sql docker git microservices",
        "javascript react nodejs mongodb express graphql",
        "java kotlin spring postgresql kubernetes aws",
        "python django postgresql redis docker ci/cd",
        "c++ golang rust systems distributed computing",
    ],
    "data_scientist": [
        "python pandas numpy sklearn tensorflow statistics",
        "python r sql tableau statistics machine learning",
        "python pytorch deep learning nlp data analytics",
        "python spark sql hadoop data engineering pipeline",
        "python keras computer vision neural networks data science",
    ],
    "devops_engineer": [
        "docker kubernetes aws terraform jenkins ci/cd",
        "azure devops ansible jenkins pipeline monitoring",
        "aws lambda serverless terraform docker kubernetes",
        "gcp kubernetes docker jenkins terraform ansible",
        "linux docker kubernetes aws monitoring automation",
    ],
    "frontend_developer": [
        "javascript react typescript html css webpack",
        "javascript angular typescript bootstrap responsive design",
        "javascript vue nodejs html css graphql",
        "react nextjs typescript tailwind frontend testing",
        "javascript react redux html css performance",
    ],
    "ml_engineer": [
        "python tensorflow keras deep learning neural networks",
        "python pytorch nlp transformers machine learning",
        "python sklearn pandas data science statistics ml",
        "python computer vision opencv tensorflow deployment",
        "python reinforcement learning pytorch ai optimization",
    ],
}

RESUME_TEMPLATES = [
    "experienced {role} with {exp} years of experience. proficient in {skills}. {edu} degree from {uni}. worked at {org}. built multiple projects.",
    "{edu} graduate with {exp} years experience in {skills}. previously worked at {org} as {role}. strong problem solving and teamwork skills.",
    "skilled {role} with expertise in {skills}. {exp} years of professional experience. {edu} from {uni}. developed applications at {org}.",
    "{role} with {exp} years experience. technical skills include {skills}. {edu} degree. experience at {org}. leadership and communication skills.",
    "passionate {role} skilled in {skills}. {exp} years in industry. {edu} from {uni}. contributed to projects at {org}. agile methodology.",
]

RESUME_ORGS = ["google", "microsoft", "amazon", "infosys", "tcs", "wipro", "accenture", "ibm", "oracle", "startup"]
RESUME_UNIS = ["mit", "stanford", "iit", "nit", "bits", "anna university", "vtu", "delhi university"]
EDUCATION_LEVELS = ["bachelor", "master", "phd", "btech", "mtech"]
ROLES = ["software engineer", "developer", "data scientist", "devops engineer", "frontend developer", "ml engineer", "analyst"]


def generate_siamese_dataset(num_pairs=3000, seed=42):
    """
    Generate resume-JD pairs with match labels.
    
    Matching pairs: Resume and JD share the same role category.
    Non-matching pairs: Resume and JD are from different role categories.
    
    Returns:
        resume_texts: list of resume text strings
        jd_texts: list of JD text strings  
        labels: list of floats (1.0 = match, 0.0 = no match)
    """
    random.seed(seed)
    
    resume_texts = []
    jd_texts = []
    labels = []
    
    role_keys = list(JD_TEMPLATES.keys())
    
    for i in range(num_pairs):
        if random.random() < 0.5:
            # MATCHING pair (label=1): same role category
            role = random.choice(role_keys)
            
            # Generate JD
            jd_template = random.choice(JD_TEMPLATES[role])
            skills = random.choice(ROLE_SKILLS[role])
            exp = str(random.randint(1, 8))
            edu = random.choice(EDUCATION_LEVELS)
            jd = jd_template.format(skills=skills, exp=exp, edu=edu)
            
            # Generate resume for same role
            resume_template = random.choice(RESUME_TEMPLATES)
            role_name = random.choice(ROLES[:2]) if role == "software_engineer" else role.replace("_", " ")
            # Use skills from the same role category (with some variation)
            resume_skills = random.choice(ROLE_SKILLS[role])
            resume_exp = str(max(1, int(exp) + random.randint(-2, 2)))
            resume = resume_template.format(
                role=role_name, skills=resume_skills, exp=resume_exp,
                edu=random.choice(EDUCATION_LEVELS),
                uni=random.choice(RESUME_UNIS),
                org=random.choice(RESUME_ORGS)
            )
            
            resume_texts.append(resume)
            jd_texts.append(jd)
            labels.append(1.0)
        else:
            # NON-MATCHING pair (label=0): different role categories
            role_jd = random.choice(role_keys)
            role_resume = random.choice([r for r in role_keys if r != role_jd])
            
            # Generate JD for one role
            jd_template = random.choice(JD_TEMPLATES[role_jd])
            skills = random.choice(ROLE_SKILLS[role_jd])
            exp = str(random.randint(1, 8))
            edu = random.choice(EDUCATION_LEVELS)
            jd = jd_template.format(skills=skills, exp=exp, edu=edu)
            
            # Generate resume for a DIFFERENT role
            resume_template = random.choice(RESUME_TEMPLATES)
            role_name = role_resume.replace("_", " ")
            resume_skills = random.choice(ROLE_SKILLS[role_resume])
            resume_exp = str(random.randint(1, 8))
            resume = resume_template.format(
                role=role_name, skills=resume_skills, exp=resume_exp,
                edu=random.choice(EDUCATION_LEVELS),
                uni=random.choice(RESUME_UNIS),
                org=random.choice(RESUME_ORGS)
            )
            
            resume_texts.append(resume)
            jd_texts.append(jd)
            labels.append(0.0)
    
    return resume_texts, jd_texts, labels


# ============================================================
# Training Function
# ============================================================
def train_siamese_model(num_pairs=3000, epochs=30, lr=0.001, save_path="siamese_model.pth"):
    """
    Train the Siamese Neural Network on synthetic resume-JD pairs.
    
    Args:
        num_pairs: Number of resume-JD pairs to generate
        epochs: Training epochs
        lr: Learning rate
        save_path: Path to save trained model
    
    Returns:
        model: Trained SiameseNetwork
        vectorizer: TextVectorizer instance
    """
    print("=" * 60)
    print("  Training Siamese Neural Network (Resume-JD Matching)")
    print("=" * 60)
    
    # Generate data
    print(f"\n[1/3] Generating {num_pairs} resume-JD pairs...")
    resume_texts, jd_texts, labels = generate_siamese_dataset(num_pairs)
    
    match_count = sum(1 for l in labels if l == 1.0)
    print(f"      Matching pairs: {match_count}")
    print(f"      Non-matching pairs: {num_pairs - match_count}")
    
    # Vectorize texts
    vectorizer = TextVectorizer()
    print(f"      Vocabulary size: {vectorizer.vector_size}")
    
    resume_vectors = vectorizer.vectorize_batch(resume_texts)
    jd_vectors = vectorizer.vectorize_batch(jd_texts)
    labels_array = np.array(labels, dtype=np.float32)
    
    # Convert to tensors
    resume_tensor = torch.FloatTensor(resume_vectors)
    jd_tensor = torch.FloatTensor(jd_vectors)
    labels_tensor = torch.FloatTensor(labels_array)
    
    # Train/validation split (85/15)
    split = int(0.85 * num_pairs)
    train_resume, val_resume = resume_tensor[:split], resume_tensor[split:]
    train_jd, val_jd = jd_tensor[:split], jd_tensor[split:]
    train_labels, val_labels = labels_tensor[:split], labels_tensor[split:]
    
    # Initialize model
    print(f"\n[2/3] Initializing Siamese Network...")
    model = SiameseNetwork(input_dim=vectorizer.vector_size, embedding_dim=64)
    criterion = ContrastiveLoss(margin=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"      Total parameters: {total_params:,}")
    print(f"      Architecture: Input({vectorizer.vector_size}) → Shared MLP → Embedding(64)")
    print(f"      Loss: Contrastive Loss (margin=2.0)")
    
    # Training loop
    print(f"\n[3/3] Training for {epochs} epochs...")
    batch_size = 64
    
    for epoch in range(epochs):
        model.train()
        
        # Shuffle training data
        perm = torch.randperm(split)
        train_resume_shuffled = train_resume[perm]
        train_jd_shuffled = train_jd[perm]
        train_labels_shuffled = train_labels[perm]
        
        total_loss = 0.0
        num_batches = 0
        
        for i in range(0, split, batch_size):
            batch_resume = train_resume_shuffled[i:i+batch_size]
            batch_jd = train_jd_shuffled[i:i+batch_size]
            batch_labels = train_labels_shuffled[i:i+batch_size]
            
            optimizer.zero_grad()
            emb_resume, emb_jd = model(batch_resume, batch_jd)
            loss = criterion(emb_resume, emb_jd, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        avg_loss = total_loss / num_batches
        
        # Validation
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                val_emb_r, val_emb_j = model(val_resume, val_jd)
                val_loss = criterion(val_emb_r, val_emb_j, val_labels)
                
                # Compute accuracy: matching pairs should have small distance
                distances = F.pairwise_distance(val_emb_r, val_emb_j)
                predictions = (distances < 1.0).float()  # threshold
                correct = (predictions == val_labels).sum().item()
                val_acc = correct / len(val_labels) * 100
            
            print(f"      Epoch {epoch+1:3d}/{epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}%")
    
    # Save model
    model.eval()
    save_data = {
        "model_state_dict": model.state_dict(),
        "input_dim": vectorizer.vector_size,
        "embedding_dim": 64,
        "vocabulary": KEYWORD_VOCABULARY,
    }
    torch.save(save_data, save_path)
    print(f"\n      Model saved to {save_path}")
    print("=" * 60)
    
    return model, vectorizer


# ============================================================
# Inference: Compute Resume-JD Similarity
# ============================================================
def load_siamese_model(model_path="siamese_model.pth"):
    """Load a trained Siamese model from disk."""
    data = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    model = SiameseNetwork(input_dim=data["input_dim"], embedding_dim=data["embedding_dim"])
    model.load_state_dict(data["model_state_dict"])
    model.eval()
    vectorizer = TextVectorizer(data["vocabulary"])
    return model, vectorizer


def compute_similarity(resume_text, jd_text, model, vectorizer):
    """
    Compute semantic similarity between a resume and job description.
    
    Args:
        resume_text: Raw resume text string
        jd_text: Raw job description text string
        model: Trained SiameseNetwork
        vectorizer: TextVectorizer instance
    
    Returns:
        similarity_score: float between 0.0 and 1.0 (1.0 = perfect match)
    """
    resume_vec = vectorizer.vectorize(resume_text)
    jd_vec = vectorizer.vectorize(jd_text)
    
    resume_tensor = torch.FloatTensor(resume_vec).unsqueeze(0)
    jd_tensor = torch.FloatTensor(jd_vec).unsqueeze(0)
    
    similarity = model.get_similarity(resume_tensor, jd_tensor)
    return similarity.item()


# ============================================================
# Main (standalone test)
# ============================================================
if __name__ == "__main__":
    model, vectorizer = train_siamese_model(num_pairs=3000, epochs=30, save_path="siamese_model.pth")
    
    # Test matching pair
    resume = "experienced python developer with 5 years experience in machine learning tensorflow keras data science"
    jd_match = "seeking data scientist with expertise in python tensorflow machine learning. 4 years experience required."
    jd_nomatch = "hiring frontend developer skilled in javascript react angular html css. 3 years experience."
    
    sim_match = compute_similarity(resume, jd_match, model, vectorizer)
    sim_nomatch = compute_similarity(resume, jd_nomatch, model, vectorizer)
    
    print(f"\nResume: {resume[:80]}...")
    print(f"Matching JD similarity:     {sim_match:.4f}")
    print(f"Non-matching JD similarity: {sim_nomatch:.4f}")
