# AI Recruitment Simulation & Intelligent Candidate Matching System
## Comprehensive Technical Documentation

---

## 1. Project Overview & Architecture

This project implements an end-to-end AI recruitment system composed of two complementary subsystems:

1. **Custom Neural Networks for Intelligent Candidate Matching (Advanced Pipeline)**:
   - **Network 1: Bi-LSTM + CRF** — Sequence labeling / Named Entity Recognition (NER) to extract skills, organizations, education, dates, and experience from resumes.
   - **Network 2: Siamese Neural Network** — Twin-branch neural network with shared weights trained via Contrastive Loss to compute semantic similarity between candidate resumes and job descriptions.
   - **Network 3: Deep Ranking Network MLP** — Deep ranking multi-layer perceptron taking the Siamese similarity score and NER-extracted features to compute final candidate-job match percentages and rank applicants on a leaderboard.

2. **NumPy Baseline Screening Network (First-Principles Implementation)**:
   - Built purely with Python and NumPy from scratch (no high-level ML framework).
   - Demonstrates foundational neural network concepts: forward propagation, backpropagation with the chain rule, weight initialization, and binary classification.

---

## 2. Directory Layout & File Overview

```text
NN mini proj/
│
├── sample_resumes/              # PDF resumes for testing
│   ├── strong_candidate.pdf     # High-skill, high-experience candidate
│   └── weak_candidate.pdf       # Low-skill, entry-level candidate
│
├── bilstm_crf_model.py          # Network 1: Custom Bi-LSTM + CRF NER from scratch
├── siamese_model.py             # Network 2: Siamese Network with Contrastive Loss
├── deep_ranking_model.py        # Network 3: Deep Ranking Network MLP + Ranking Loss
├── inference_pipeline.py        # End-to-end inference connecting all 3 neural networks
├── train_all_models.py          # Unified CPU-optimized training script for all 3 models
├── test_pdf.py                  # Verification test for PDF parsing + 3-network inference
│
├── ner_model.pth                # Trained weights + vocabulary for Bi-LSTM + CRF
├── siamese_model.pth            # Trained weights + vectorizer for Siamese Network
├── ranking_model.pth            # Trained weights + normalizer for Deep Ranking MLP
│
├── app.py                       # Enhanced Streamlit GUI (Tab 1: Neural Matching, Tab 2: PDF, Tab 3: Manual)
├── model_base.py                # Pure NumPy baseline neural network class
├── train.py                     # Baseline NumPy training script
├── generate_data.py             # Baseline synthetic CSV dataset generator
├── resume_parser.py             # PDF text extraction & regex parser
├── verify_pipeline.py           # Baseline 5-phase testing suite
├── train_candidates.csv         # 450 synthetic baseline training records
├── test_candidates.csv          # 100 synthetic baseline test records
├── model_weights.npz            # Saved weights for NumPy baseline model
├── implementation_progress.txt  # Implementation milestone tracker
└── Project_Documentation.md     # This comprehensive technical guide
```

---

## 3. Network 1: Bi-LSTM + CRF (Named Entity Recognition)

### Purpose
To parse unstructured text from candidate resumes and extract structured entities:
- **Skills** (e.g., Python, TensorFlow, Machine Learning, React, AWS)
- **Organizations** (e.g., Google, Infosys, Stanford, MIT)
- **Education** (e.g., Bachelor, Master, PhD, Computer Science)
- **Dates** (e.g., 2021, 2023, January 2020)
- **Experience** (e.g., 5 years, 6 months)

### Architecture
```text
Input Tokens
     ↓
Embedding Layer (vocab_size, 64)
     ↓
Dropout (p = 0.3)
     ↓
Bidirectional LSTM (2 layers, hidden_dim = 128 // 2 = 64 per direction)
  - Forward LSTM captures past context
  - Backward LSTM captures future context
     ↓
Linear Projection (128 → 13 tag classes)
     ↓
Conditional Random Field (CRF) Layer
  - Models transition probabilities between adjacent tags
  - Prevents illegal sequences (e.g., I-SKILL after O)
     ↓
Viterbi Decoding → Optimal Tag Sequence
```

### Mathematical Formulation
The CRF models the conditional probability of a tag sequence $\mathbf{y} = (y_1, y_2, \dots, y_n)$ given token emissions $\mathbf{x}$:

$$P(\mathbf{y} \mid \mathbf{x}) = \frac{\exp(\text{Score}(\mathbf{x}, \mathbf{y}))}{\sum_{\mathbf{y}'} \exp(\text{Score}(\mathbf{x}, \mathbf{y}'))} = \frac{\exp(\text{Score}(\mathbf{x}, \mathbf{y}))}{Z(\mathbf{x})}$$

Where the sequence score is the sum of emission and transition scores:

$$\text{Score}(\mathbf{x}, \mathbf{y}) = \sum_{i=1}^n P_{i, y_i} + \sum_{i=0}^n A_{y_i, y_{i+1}}$$

- $P_{i, j}$: Emission score of token $i$ being assigned tag $j$ from the Bi-LSTM output.
- $A_{j, k}$: Transition score of transitioning from tag $j$ to tag $k$.
- $Z(\mathbf{x})$: Partition function, computed efficiently using the **Forward Algorithm** via log-sum-exp dynamic programming.

**Training Objective**: Minimize Negative Log-Likelihood (NLL):
$$\mathcal{L}_{\text{CRF}} = -\log P(\mathbf{y}_{\text{gold}} \mid \mathbf{x}) = \log Z(\mathbf{x}) - \text{Score}(\mathbf{x}, \mathbf{y}_{\text{gold}})$$

**Inference**: **Viterbi Algorithm** dynamic programming finds the global highest-scoring tag sequence $\mathbf{y}^* = \arg\max_{\mathbf{y}} \text{Score}(\mathbf{x}, \mathbf{y})$.

---

## 4. Network 2: Siamese Neural Network (Resume–JD Semantic Matching)

### Purpose
To project candidate resumes and job descriptions into a shared semantic latent space, allowing the system to compute their semantic alignment using distance/similarity metrics.

### Twin Architecture with Shared Weights
```text
   Resume Text                      Job Description Text
        ↓                                    ↓
Text Vectorizer (Bag-of-Words)        Text Vectorizer (Bag-of-Words)
        ↓                                    ↓
┌──────────────────┐                 ┌──────────────────┐
│ Branch A (MLP)   │                 │ Branch B (MLP)   │
│ Linear(139 → 128)│                 │ Linear(139 → 128)│
│ BatchNorm + ReLU │   SHARED        │ BatchNorm + ReLU │
│ Dropout(0.2)     │   WEIGHTS       │ Dropout(0.2)     │
│ Linear(128 → 64) │   (W_A = W_B)   │ Linear(128 → 64) │
│ L2 Normalization │                 │ L2 Normalization │
└──────────────────┘                 └──────────────────┘
        ↓                                    ↓
Latent Vector u (64-dim)             Latent Vector v (64-dim)
        └─────────────────┬──────────────────┘
                          ↓
              Cosine Similarity / Distance
              d(u, v) = ||u - v||_2
```

### Contrastive Loss Formulation
Trained with **Hadsell et al. Contrastive Loss**:

$$\mathcal{L}_{\text{Contrastive}}(y, d) = \frac{1}{2} y \cdot d^2 + \frac{1}{2} (1 - y) \cdot \max(0, m - d)^2$$

Where:
- $y \in \{0, 1\}$: Ground-truth label ($1$ = matching pair, $0$ = non-matching pair).
- $d = \|\mathbf{u} - \mathbf{v}\|_2$: Euclidean distance between normalized latent vectors.
- $m$: Margin hyperparameter ($m = 2.0$), penalizing non-matching pairs that are closer than distance $m$.

**Semantic Similarity Score**:
$$\text{Sim}(\mathbf{u}, \mathbf{v}) = \frac{1}{1 + d(\mathbf{u}, \mathbf{v})} \in (0, 1]$$

---

## 5. Network 3: Deep Ranking Network MLP (Final Match Scoring)

### Purpose
Combines the holistic semantic similarity score with structured candidate attributes (extracted by the Bi-LSTM+CRF) to predict a final match percentage and rank candidates on a leaderboard.

### Input Vector (6 Dimensions)
1. **Siamese Similarity**: Output from Network 2 ($0.0 - 1.0$)
2. **Skill Count**: Number of unique domain skills extracted from resume
3. **Experience (Years)**: Numeric total years of experience
4. **Education Level**: Encoded hierarchy (0 = None, 1 = Diploma, 2 = Bachelor, 3 = Master, 4 = PhD)
5. **Organization Count**: Reputable previous employers/institutions
6. **Project Count**: Volume of completed technical projects

### Architecture
```text
Input Feature Vector (6 dimensions)
     ↓
Linear(6 → 128) + BatchNorm + ReLU + Dropout(0.2)
     ↓
Linear(128 → 64) + BatchNorm + ReLU + Dropout(0.2)
     ↓
Linear(64 → 32) + ReLU
     ↓
Linear(32 → 1) + Sigmoid Activation
     ↓
Final Match Probability ∈ [0, 1]  (Scaled to 0 - 100%)
```

### Hybrid Ranking Loss
Trained with a composite loss function combining point-wise classification and pair-wise ranking:

$$\mathcal{L}_{\text{Total}} = \alpha \cdot \mathcal{L}_{\text{BCE}} + (1 - \alpha) \cdot \mathcal{L}_{\text{MarginRanking}}$$

- $\mathcal{L}_{\text{BCE}} = - [y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})]$
- $\mathcal{L}_{\text{MarginRanking}} = \max(0, -y_{\text{pair}} \cdot (\hat{y}_1 - \hat{y}_2) + \text{margin})$
- $\alpha = 0.6$ balances calibrated probability output with pairwise ordering.

---

## 6. End-to-End Pipeline & User Interface

### Pipeline Flow (`inference_pipeline.py`)
```text
Candidate Resume (PDF or Raw Text) + Job Description
                      ↓
  [Step 1] Bi-LSTM + CRF NER:
           Extracts Skills, Organizations, Education, Dates, Experience
                      ↓
  [Step 2] Siamese Neural Network:
           Computes semantic embedding similarity between Resume and JD
                      ↓
  [Step 3] Deep Ranking MLP:
           Takes [Siamese Score, Skill Count, Exp, Edu Level, Orgs, Projects]
                      ↓
  Final Output:
  • Decision: SHORTLISTED (>= 50%) or REJECTED (< 50%)
  • Match Percentage & Confidence Score
  • Full entity breakdown & feature table
  • Leaderboard rank among competing applicants
```

### Streamlit Application (`app.py`)
- **Tab 1: Intelligent Matching (3 Neural Networks)**
  - Dual-panel UI: Job Description input on left, Resume input (PDF upload or text paste) on right.
  - Live prediction card with decision badge, match score gauge, and confidence metric.
  - Multi-column visual breakdown:
    - Column 1: Bi-LSTM + CRF extracted entities.
    - Column 2: Siamese Network semantic similarity bar with descriptive match classification.
    - Column 3: Deep Ranking MLP feature table and score.
  - Batch Candidate Comparison: Compare 2 to 10 candidates simultaneously against one JD on a ranked leaderboard.
- **Tab 2: Resume Screening (PDF)**
  - Original single-PDF screening forward pass with under-the-hood matrix math inspection.
- **Tab 3: Manual Simulation**
  - Sliders for experience, technical score, soft skills, and projects, plus batch testing leaderboard.

---

## 7. Verification & Testing Evidence

All modules include automated verification tests:
1. `python train_all_models.py` — Trains and saves all 3 models in ~2.5 minutes on CPU.
2. `python inference_pipeline.py` — Tests Alice (Senior Data Scientist match), Bob (Web Dev), and Charlie (Frontend) on a Senior Data Scientist role:
   - Alice: **79.1% match (SHORTLISTED)**
   - Bob: **10.9% match (REJECTED)**
   - Charlie: **16.7% match (REJECTED)**
3. `python test_pdf.py` — Tests PDF extraction and neural scoring:
   - `strong_candidate.pdf`: **99.7% match (SHORTLISTED)**
   - `weak_candidate.pdf`: **24.6% match (REJECTED)**
4. `python verify_pipeline.py` — Full 5-phase test suite for baseline dataset, weights, and logic.

---

## 8. Summary for Project Viva / Defense

Key points to present during evaluation:
1. **Why Bi-LSTM + CRF instead of standard NER?**
   The Bi-LSTM captures bidirectional contextual token representations, while the CRF explicitly learns valid tag transition dependencies (preventing syntactically invalid sequences like an `I-SKILL` without a prior `B-SKILL`).
2. **Why a Siamese Network with Contrastive Loss?**
   Standard classifiers fail when matching arbitrary text pairs because resumes and JDs have different lengths and wording. Siamese networks learn a shared metric space where similar semantic pairs are pulled together and dissimilar pairs are pushed apart by at least margin $m$.
3. **Why Deep Ranking MLP on top of Siamese?**
   Semantic similarity alone doesn't measure seniority or qualification fit. A junior developer with Python skills has high semantic similarity to a Senior Architect role, but lacks experience years and education credentials. The Deep Ranking MLP integrates both semantic and structured factors.
