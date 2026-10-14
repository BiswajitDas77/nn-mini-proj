"""
Deep Ranking Network MLP for Final Candidate–Job Match Scoring
==============================================================
Takes the Siamese similarity score along with NER-extracted features
(skills count, experience years, education level, organization count)
to predict the final candidate–job match score.

Architecture:
    Input Vector = [siamese_similarity, skill_count, experience_years, 
                    education_level, org_count, project_count]
    
    → Linear(6, 128) → BatchNorm → ReLU → Dropout(0.3)
    → Linear(128, 64) → BatchNorm → ReLU → Dropout(0.2)
    → Linear(64, 32) → ReLU
    → Linear(32, 1) → Sigmoid
    
    Output: Match score between 0.0 and 1.0

This network learns to weight and combine the semantic similarity from
the Siamese network with structured features from the BiLSTM-CRF NER
to produce a final calibrated match probability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


# ============================================================
# Deep Ranking Network MLP
# ============================================================
class DeepRankingMLP(nn.Module):
    """
    Deep Ranking Multi-Layer Perceptron.
    
    Combines semantic similarity (from Siamese Network) with
    structured HR features (from BiLSTM-CRF NER) to produce
    a final candidate-job match score.
    
    Input features (6 dimensions):
        [0] siamese_similarity  - Semantic match score (0-1)
        [1] skill_count         - Number of relevant skills found by NER
        [2] experience_years    - Years of experience extracted by NER
        [3] education_level     - Encoded education level (0-4)
        [4] org_count           - Number of organizations/companies
        [5] project_count       - Number of projects mentioned
    
    Output:
        match_score: float in [0, 1], probability of candidate-job fit
    """
    
    def __init__(self, input_dim=6):
        super(DeepRankingMLP, self).__init__()
        
        self.input_dim = input_dim
        
        self.network = nn.Sequential(
            # Layer 1: Expand feature space
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Layer 2: Feature interaction learning
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Layer 3: Compression
            nn.Linear(64, 32),
            nn.ReLU(),
            
            # Output layer: Match probability
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights using Kaiming for ReLU layers
        self._init_weights()
    
    def _init_weights(self):
        """Apply He initialization for ReLU and Xavier for Sigmoid."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                if module == self.network[-2]:
                    # Last linear layer before sigmoid → Xavier
                    nn.init.xavier_uniform_(module.weight)
                else:
                    # ReLU layers → He/Kaiming
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (batch, input_dim) feature vectors
        Returns:
            (batch, 1) match scores
        """
        return self.network(x)
    
    def predict_score(self, features):
        """
        Predict match score for a single candidate.
        
        Args:
            features: numpy array or list of input features
        Returns:
            float: match score between 0.0 and 1.0
        """
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(features).unsqueeze(0) if not isinstance(features, torch.Tensor) else features
            if x.dim() == 1:
                x = x.unsqueeze(0)
            score = self.forward(x)
        return score.squeeze().item()


# ============================================================
# Ranking Loss (Margin Ranking Loss for Pairwise Learning)
# ============================================================
class RankingLoss(nn.Module):
    """
    Combined loss for the ranking network:
        1. BCE Loss: For calibrated probability prediction
        2. Margin Ranking Loss: To ensure correct relative ordering
        
    Total Loss = α · BCE + (1-α) · MarginRanking
    
    The Margin Ranking Loss enforces that better candidates
    score higher than worse candidates:
        L_rank = max(0, -sign(y_i - y_j) · (s_i - s_j) + margin)
    """
    
    def __init__(self, alpha=0.6, margin=0.1):
        super(RankingLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCELoss()
        self.margin_ranking = nn.MarginRankingLoss(margin=margin)
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (batch, 1) predicted match scores
            targets: (batch, 1) true match scores/labels
        """
        # BCE component
        bce_loss = self.bce(predictions, targets)
        
        # Margin Ranking component (pairwise within batch)
        batch_size = predictions.size(0)
        if batch_size < 2:
            return bce_loss
        
        # Create pairs
        idx = torch.randperm(batch_size)
        pred_i = predictions[idx[:batch_size//2]]
        pred_j = predictions[idx[batch_size//2:batch_size//2*2]]
        target_i = targets[idx[:batch_size//2]]
        target_j = targets[idx[batch_size//2:batch_size//2*2]]
        
        # Sign: +1 if target_i > target_j, -1 otherwise
        sign = torch.sign(target_i - target_j)
        # Replace zeros with +1
        sign[sign == 0] = 1.0
        
        ranking_loss = self.margin_ranking(pred_i, pred_j, sign)
        
        total_loss = self.alpha * bce_loss + (1 - self.alpha) * ranking_loss
        return total_loss


# ============================================================
# Feature Normalization
# ============================================================
class FeatureNormalizer:
    """
    Min-Max normalizer for input features.
    Stores min/max values for consistent normalization during inference.
    """
    
    def __init__(self):
        self.min_vals = None
        self.max_vals = None
        self.fitted = False
    
    def fit(self, X):
        """Compute min/max from training data."""
        self.min_vals = np.min(X, axis=0)
        self.max_vals = np.max(X, axis=0)
        # Prevent division by zero
        range_vals = self.max_vals - self.min_vals
        range_vals[range_vals == 0] = 1e-8
        self.range_vals = range_vals
        self.fitted = True
    
    def transform(self, X):
        """Normalize features to [0, 1]."""
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before transform")
        return (X - self.min_vals) / self.range_vals
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)
    
    def to_dict(self):
        return {"min_vals": self.min_vals.tolist(), "max_vals": self.max_vals.tolist(), "range_vals": self.range_vals.tolist()}
    
    @classmethod
    def from_dict(cls, d):
        n = cls()
        n.min_vals = np.array(d["min_vals"])
        n.max_vals = np.array(d["max_vals"])
        n.range_vals = np.array(d["range_vals"])
        n.fitted = True
        return n


# ============================================================
# Synthetic Ranking Dataset Generator
# ============================================================
def generate_ranking_dataset(num_samples=4000, seed=42):
    """
    Generate synthetic training data for the ranking network.
    
    Each sample: [siamese_sim, skill_count, exp_years, edu_level, org_count, proj_count] → match_score
    
    The match_score is computed using a weighted formula with noise injection
    to simulate real-world subjectivity in hiring decisions.
    
    Returns:
        features: (num_samples, 6) numpy array
        labels: (num_samples, 1) numpy array of match scores
    """
    random.seed(seed)
    np.random.seed(seed)
    
    features = []
    labels = []
    
    for _ in range(num_samples):
        # Generate random features
        siamese_sim = np.random.uniform(0.1, 0.95)        # Siamese similarity score
        skill_count = np.random.randint(0, 15)             # Number of relevant skills
        exp_years = np.random.uniform(0, 10)               # Years of experience
        edu_level = np.random.randint(0, 5)                # 0=none, 1=diploma, 2=bachelor, 3=master, 4=phd
        org_count = np.random.randint(0, 6)                # Number of organizations
        proj_count = np.random.randint(0, 8)               # Number of projects
        
        feature_vec = [siamese_sim, skill_count, exp_years, edu_level, org_count, proj_count]
        
        # Compute target match score using weighted formula
        # Higher weight on semantic similarity and skills
        match_score = (
            0.30 * siamese_sim +                           # Semantic match is most important
            0.25 * min(skill_count / 10.0, 1.0) +          # Skill coverage
            0.20 * min(exp_years / 8.0, 1.0) +             # Experience
            0.12 * (edu_level / 4.0) +                     # Education
            0.05 * min(org_count / 4.0, 1.0) +             # Work history
            0.08 * min(proj_count / 5.0, 1.0)              # Project experience
        )
        
        # Inject noise to simulate subjectivity (±5%)
        noise = np.random.normal(0, 0.05)
        match_score = np.clip(match_score + noise, 0.0, 1.0)
        
        # Binary classification with soft threshold
        # Candidates with match_score > 0.55 are considered a match
        if match_score > 0.55:
            label = 1.0
        elif match_score < 0.40:
            label = 0.0
        else:
            # Borderline candidates: use match_score as soft label
            label = match_score
        
        # Random label flip (3%) for realism
        if random.random() < 0.03:
            label = 1.0 - label if label in [0.0, 1.0] else label
        
        features.append(feature_vec)
        labels.append([label])
    
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.float32)


# ============================================================
# Training Function
# ============================================================
def train_ranking_model(num_samples=4000, epochs=40, lr=0.001, save_path="ranking_model.pth"):
    """
    Train the Deep Ranking Network MLP.
    
    Args:
        num_samples: Number of training samples to generate
        epochs: Training epochs
        lr: Learning rate
        save_path: Path to save trained model
    
    Returns:
        model: Trained DeepRankingMLP
        normalizer: FeatureNormalizer instance
    """
    print("=" * 60)
    print("  Training Deep Ranking Network MLP")
    print("=" * 60)
    
    # Generate data
    print(f"\n[1/3] Generating {num_samples} ranking samples...")
    features, labels = generate_ranking_dataset(num_samples)
    
    positive = np.sum(labels > 0.5)
    print(f"      Feature dimensions: {features.shape[1]}")
    print(f"      Positive matches: {int(positive)}")
    print(f"      Negative matches: {num_samples - int(positive)}")
    
    # Normalize features
    normalizer = FeatureNormalizer()
    features_norm = normalizer.fit_transform(features)
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(features_norm)
    Y_tensor = torch.FloatTensor(labels)
    
    # Train/validation split (85/15)
    split = int(0.85 * num_samples)
    X_train, X_val = X_tensor[:split], X_tensor[split:]
    Y_train, Y_val = Y_tensor[:split], Y_tensor[split:]
    
    # Initialize model
    print(f"\n[2/3] Initializing Deep Ranking MLP...")
    model = DeepRankingMLP(input_dim=features.shape[1])
    criterion = RankingLoss(alpha=0.6, margin=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"      Total parameters: {total_params:,}")
    print(f"      Architecture: Input(6) → 128 → 64 → 32 → 1 (Sigmoid)")
    print(f"      Loss: BCE + Margin Ranking (α=0.6)")
    
    # Training loop
    print(f"\n[3/3] Training for {epochs} epochs...")
    batch_size = 64
    
    for epoch in range(epochs):
        model.train()
        
        # Shuffle
        perm = torch.randperm(split)
        X_shuffled = X_train[perm]
        Y_shuffled = Y_train[perm]
        
        total_loss = 0.0
        num_batches = 0
        
        for i in range(0, split, batch_size):
            batch_x = X_shuffled[i:i+batch_size]
            batch_y = Y_shuffled[i:i+batch_size]
            
            if batch_x.size(0) < 2:
                continue
            
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)
        
        # Validation
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val)
                val_loss = criterion(val_preds, Y_val)
                
                # Accuracy (binary threshold at 0.5)
                binary_preds = (val_preds > 0.5).float()
                binary_labels = (Y_val > 0.5).float()
                correct = (binary_preds == binary_labels).sum().item()
                val_acc = correct / len(Y_val) * 100
            
            print(f"      Epoch {epoch+1:3d}/{epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}%")
    
    # Save model
    model.eval()
    save_data = {
        "model_state_dict": model.state_dict(),
        "input_dim": features.shape[1],
        "normalizer": normalizer.to_dict(),
    }
    torch.save(save_data, save_path)
    print(f"\n      Model saved to {save_path}")
    print("=" * 60)
    
    return model, normalizer


# ============================================================
# Inference
# ============================================================
def load_ranking_model(model_path="ranking_model.pth"):
    """Load a trained ranking model from disk."""
    data = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    model = DeepRankingMLP(input_dim=data["input_dim"])
    model.load_state_dict(data["model_state_dict"])
    model.eval()
    normalizer = FeatureNormalizer.from_dict(data["normalizer"])
    return model, normalizer


def rank_candidates(candidates_features, model, normalizer):
    """
    Rank a list of candidates by their match scores.
    
    Args:
        candidates_features: list of feature dicts or numpy array (N, 6)
            Each: [siamese_sim, skill_count, exp_years, edu_level, org_count, proj_count]
        model: Trained DeepRankingMLP
        normalizer: FeatureNormalizer
    
    Returns:
        List of (index, match_score) tuples sorted by score descending
    """
    if isinstance(candidates_features, list):
        features = np.array(candidates_features, dtype=np.float32)
    else:
        features = candidates_features.astype(np.float32)
    
    features_norm = normalizer.transform(features)
    
    model.eval()
    with torch.no_grad():
        X = torch.FloatTensor(features_norm)
        scores = model(X).squeeze().numpy()
    
    if scores.ndim == 0:
        scores = np.array([scores])
    
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked


# ============================================================
# Main (standalone test)
# ============================================================
if __name__ == "__main__":
    model, normalizer = train_ranking_model(num_samples=4000, epochs=40, save_path="ranking_model.pth")
    
    # Test ranking
    test_candidates = [
        [0.9, 12, 7.0, 3, 4, 6],  # Strong match
        [0.3, 3, 1.0, 1, 1, 1],   # Weak match
        [0.7, 8, 4.5, 2, 2, 3],   # Moderate match
        [0.5, 5, 2.0, 2, 1, 2],   # Below average
    ]
    
    ranked = rank_candidates(np.array(test_candidates), model, normalizer)
    print("\nCandidate Rankings:")
    for rank, (idx, score) in enumerate(ranked, 1):
        print(f"  #{rank}: Candidate {idx+1} → Match Score: {score:.4f}")
