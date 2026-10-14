"""
Bi-LSTM + CRF Model for Resume Named Entity Recognition
========================================================
Identifies: SKILL, ORG (Organization), EDU (Education), DATE, EXP (Experience)
Uses BIO tagging scheme (Begin, Inside, Outside).

Architecture:
    Word Embedding → Bidirectional LSTM → Linear Projection → CRF Layer
    
The CRF layer is implemented from scratch (no external library) to enforce
valid tag transition constraints during decoding.
"""

import torch
import torch.nn as nn
import numpy as np
import random
import json
import os
import re

# ============================================================
# Tag Definitions (BIO Scheme)
# ============================================================
TAG_TO_IX = {
    "O": 0,
    "B-SKILL": 1, "I-SKILL": 2,
    "B-ORG": 3, "I-ORG": 4,
    "B-EDU": 5, "I-EDU": 6,
    "B-DATE": 7, "I-DATE": 8,
    "B-EXP": 9, "I-EXP": 10,
    "<START>": 11, "<STOP>": 12
}
IX_TO_TAG = {v: k for k, v in TAG_TO_IX.items()}
NUM_TAGS = len(TAG_TO_IX)

# ============================================================
# CRF Layer (Implemented from Scratch)
# ============================================================
class CRF(nn.Module):
    """
    Conditional Random Field layer.
    
    Unlike a simple softmax classifier that makes independent predictions
    per token, CRF models the joint probability of the entire tag sequence,
    enforcing valid transitions (e.g., I-SKILL can only follow B-SKILL or I-SKILL).
    
    Key equations:
        Score(x, y) = Σ Emission(x_i, y_i) + Σ Transition(y_{i-1}, y_i)
        P(y|x) = exp(Score(x,y)) / Σ_{y'} exp(Score(x,y'))
        Loss = -log P(y|x) = log(Σ exp(Score(x,y'))) - Score(x,y)
    """
    
    def __init__(self, num_tags):
        super(CRF, self).__init__()
        self.num_tags = num_tags
        
        # Transition scores: transitions[i][j] = score of transitioning FROM tag j TO tag i
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        
        # Constraint: no transitions TO <START> and no transitions FROM <STOP>
        self.transitions.data[TAG_TO_IX["<START>"], :] = -10000.0
        self.transitions.data[:, TAG_TO_IX["<STOP>"]] = -10000.0
    
    def _forward_algorithm(self, emissions):
        """
        Forward algorithm to compute the log partition function (log Z(x)).
        Uses log-sum-exp for numerical stability.
        
        Args:
            emissions: (seq_len, num_tags) emission scores from BiLSTM
        Returns:
            Log of the partition function (scalar)
        """
        seq_len = emissions.shape[0]
        
        # Initialize forward variables with START tag scores
        # alpha[j] = log of the total score of all partial paths ending in tag j
        alpha = torch.full((1, self.num_tags), -10000.0)
        alpha[0][TAG_TO_IX["<START>"]] = 0.0
        
        for t in range(seq_len):
            alpha_t = []
            for next_tag in range(self.num_tags):
                # Emission score for this tag at position t (broadcast across all prev tags)
                emit_score = emissions[t][next_tag].view(1, -1).expand(1, self.num_tags)
                # Transition score from all previous tags to next_tag
                trans_score = self.transitions[next_tag].view(1, -1)
                # Combine: previous alpha + transition + emission
                next_tag_var = alpha + trans_score + emit_score
                alpha_t.append(torch.logsumexp(next_tag_var, dim=1).view(1))
            alpha = torch.cat(alpha_t).view(1, -1)
        
        # Add transition to STOP tag
        terminal_var = alpha + self.transitions[TAG_TO_IX["<STOP>"]]
        return torch.logsumexp(terminal_var, dim=1)
    
    def _score_sentence(self, emissions, tags):
        """
        Compute the score of a given tag sequence (the numerator in P(y|x)).
        
        Args:
            emissions: (seq_len, num_tags) emission scores
            tags: (seq_len,) gold tag indices
        Returns:
            Score of the tag sequence (scalar)
        """
        score = torch.zeros(1)
        
        # Prepend START tag
        tags = torch.cat([torch.tensor([TAG_TO_IX["<START>"]], dtype=torch.long), tags])
        
        for i, emission in enumerate(emissions):
            # Transition from tags[i] to tags[i+1] + emission score at tags[i+1]
            score = score + self.transitions[tags[i + 1], tags[i]] + emission[tags[i + 1]]
        
        # Add transition to STOP tag
        score = score + self.transitions[TAG_TO_IX["<STOP>"], tags[-1]]
        return score
    
    def neg_log_likelihood(self, emissions, tags):
        """
        Negative log-likelihood loss.
        Loss = log Z(x) - Score(x, y_gold)
        """
        forward_score = self._forward_algorithm(emissions)
        gold_score = self._score_sentence(emissions, tags)
        return forward_score - gold_score
    
    def viterbi_decode(self, emissions):
        """
        Viterbi algorithm to find the best (highest-scoring) tag sequence.
        
        Args:
            emissions: (seq_len, num_tags) emission scores
        Returns:
            best_score: score of the best path
            best_path: list of tag indices
        """
        seq_len = emissions.shape[0]
        
        # Backpointers to recover the best path
        backpointers = []
        
        # Initialize with START tag
        viterbi_vars = torch.full((1, self.num_tags), -10000.0)
        viterbi_vars[0][TAG_TO_IX["<START>"]] = 0.0
        
        for t in range(seq_len):
            bptrs_t = []
            viterbivars_t = []
            
            for next_tag in range(self.num_tags):
                # For each next_tag, find which previous tag gives the best score
                next_tag_var = viterbi_vars + self.transitions[next_tag]
                best_tag_id = torch.argmax(next_tag_var)
                bptrs_t.append(best_tag_id.item())
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            
            # Add emission scores
            viterbi_vars = (torch.cat(viterbivars_t) + emissions[t]).view(1, -1)
            backpointers.append(bptrs_t)
        
        # Transition to STOP tag
        terminal_var = viterbi_vars + self.transitions[TAG_TO_IX["<STOP>"]]
        best_tag_id = torch.argmax(terminal_var).item()
        best_score = terminal_var[0][best_tag_id]
        
        # Trace back through backpointers
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        
        # Remove START tag and reverse
        best_path = best_path[:-1]
        best_path.reverse()
        
        return best_score, best_path


# ============================================================
# Bi-LSTM + CRF Model
# ============================================================
class BiLSTM_CRF(nn.Module):
    """
    Bidirectional LSTM + CRF for sequence labeling.
    
    Architecture:
        Input tokens → Embedding(vocab_size, embedding_dim)
            → BiLSTM(embedding_dim, hidden_dim)  [reads left-to-right AND right-to-left]
            → Linear(hidden_dim, num_tags)  [projects to tag space]
            → CRF(num_tags)  [enforces valid tag transitions]
    
    The BiLSTM captures contextual information from both directions,
    while the CRF layer ensures globally consistent tag sequences.
    """
    
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, num_tags=NUM_TAGS):
        super(BiLSTM_CRF, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_tags = num_tags
        
        # Word embeddings
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Bidirectional LSTM
        # hidden_dim // 2 because BiLSTM concatenates forward and backward outputs
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim // 2,
            num_layers=2,
            bidirectional=True, 
            batch_first=True,
            dropout=0.3
        )
        
        # Linear layer to project LSTM output to tag space
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # CRF layer
        self.crf = CRF(num_tags)
    
    def _get_lstm_features(self, sentence):
        """
        Get emission scores from BiLSTM for each token.
        
        Args:
            sentence: (seq_len,) tensor of word indices
        Returns:
            (seq_len, num_tags) emission scores
        """
        embeds = self.word_embeds(sentence)
        embeds = self.dropout(embeds)
        
        lstm_out, _ = self.lstm(embeds.unsqueeze(0))  # Add batch dim
        lstm_out = lstm_out.squeeze(0)  # Remove batch dim
        lstm_out = self.dropout(lstm_out)
        
        emissions = self.hidden2tag(lstm_out)
        return emissions
    
    def neg_log_likelihood(self, sentence, tags):
        """Compute negative log-likelihood for training."""
        emissions = self._get_lstm_features(sentence)
        return self.crf.neg_log_likelihood(emissions, tags)
    
    def forward(self, sentence):
        """Predict best tag sequence using Viterbi decoding."""
        emissions = self._get_lstm_features(sentence)
        score, tag_seq = self.crf.viterbi_decode(emissions)
        return score, tag_seq


# ============================================================
# Synthetic NER Dataset Generator
# ============================================================

# Entity dictionaries for data generation
SKILLS_LIST = [
    "python", "java", "javascript", "react", "nodejs", "sql", "mongodb",
    "docker", "kubernetes", "aws", "azure", "tensorflow", "pytorch",
    "pandas", "numpy", "git", "linux", "html", "css", "typescript",
    "flask", "django", "spring", "angular", "vue", "redis", "kafka",
    "spark", "hadoop", "tableau", "excel", "powerbi", "scala", "rust",
    "golang", "ruby", "swift", "kotlin", "matlab", "r"
]

MULTI_WORD_SKILLS = [
    ("machine", "learning"), ("deep", "learning"), ("data", "science"),
    ("natural", "language", "processing"), ("computer", "vision"),
    ("data", "engineering"), ("cloud", "computing"), ("web", "development"),
    ("mobile", "development"), ("software", "engineering"),
    ("data", "analysis"), ("project", "management"),
    ("agile", "methodology"), ("devops", "engineering"),
    ("artificial", "intelligence"), ("neural", "networks"),
]

ORGS = [
    "google", "microsoft", "amazon", "meta", "apple", "netflix",
    "infosys", "tcs", "wipro", "cognizant", "accenture", "deloitte",
    "ibm", "oracle", "salesforce", "adobe", "uber", "airbnb",
    "spotify", "twitter", "linkedin", "nvidia", "intel", "cisco",
    "samsung", "sony", "tesla", "spacex", "stripe", "shopify"
]

MULTI_WORD_ORGS = [
    ("goldman", "sachs"), ("morgan", "stanley"), ("jpmorgan", "chase"),
    ("bank", "of", "america"), ("wells", "fargo"),
    ("boston", "consulting"), ("mckinsey", "company"),
    ("ernst", "young"), ("price", "waterhouse"),
]

EDUCATION_TERMS = [
    "bachelor", "master", "phd", "diploma", "mba",
    "btech", "mtech", "bsc", "msc", "bca", "mca"
]

MULTI_WORD_EDU = [
    ("computer", "science"), ("information", "technology"),
    ("electrical", "engineering"), ("mechanical", "engineering"),
    ("data", "science"), ("artificial", "intelligence"),
    ("software", "engineering"), ("business", "administration"),
]

UNIVERSITIES = [
    "mit", "stanford", "harvard", "iit", "nit", "bits",
    "cambridge", "oxford", "berkeley", "caltech"
]

MULTI_WORD_UNIS = [
    ("anna", "university"), ("vtu", "university"),
    ("delhi", "university"), ("mumbai", "university"),
]

DATE_PATTERNS = [
    ("2018",), ("2019",), ("2020",), ("2021",), ("2022",), ("2023",), ("2024",), ("2025",),
    ("january", "2020"), ("february", "2021"), ("march", "2022"),
    ("june", "2023"), ("july", "2019"), ("august", "2024"),
    ("jan", "2021"), ("feb", "2022"), ("mar", "2023"), ("apr", "2024"),
]

EXP_PATTERNS = [
    ("2", "years"), ("3", "years"), ("5", "years"), ("1", "year"),
    ("6", "months"), ("18", "months"), ("4", "years"),
    ("7", "years"), ("10", "years"), ("8", "months"),
    ("1.5", "years"), ("2.5", "years"), ("3.5", "years"),
]

# Sentence templates: Each is (list_of_words_or_entity_placeholders)
# Placeholders: {SKILL}, {ORG}, {EDU}, {DATE}, {EXP}
SENTENCE_TEMPLATES = [
    # Skill mentions
    ["proficient", "in", "{SKILL}", "and", "{SKILL}"],
    ["experienced", "with", "{SKILL}", "{SKILL}", "and", "{SKILL}"],
    ["skilled", "in", "{SKILL}"],
    ["strong", "knowledge", "of", "{SKILL}", "and", "{SKILL}"],
    ["expertise", "in", "{SKILL}", "{SKILL}"],
    ["hands", "on", "experience", "with", "{SKILL}"],
    ["worked", "extensively", "with", "{SKILL}", "and", "{SKILL}"],
    ["familiar", "with", "{SKILL}", "{SKILL}", "{SKILL}"],
    ["developed", "applications", "using", "{SKILL}"],
    ["built", "systems", "with", "{SKILL}", "and", "{SKILL}"],
    
    # Organization mentions
    ["worked", "at", "{ORG}", "as", "software", "engineer"],
    ["previously", "employed", "at", "{ORG}"],
    ["internship", "at", "{ORG}", "in", "{DATE}"],
    ["senior", "developer", "at", "{ORG}"],
    ["joined", "{ORG}", "in", "{DATE}"],
    ["currently", "working", "at", "{ORG}"],
    ["software", "engineer", "at", "{ORG}", "for", "{EXP}"],
    
    # Education mentions
    ["{EDU}", "in", "{MULTI_EDU}", "from", "{UNI}"],
    ["{EDU}", "degree", "in", "{MULTI_EDU}"],
    ["completed", "{EDU}", "from", "{UNI}", "in", "{DATE}"],
    ["graduated", "with", "{EDU}", "in", "{MULTI_EDU}"],
    ["pursuing", "{EDU}", "in", "{MULTI_EDU}"],
    
    # Experience mentions
    ["{EXP}", "of", "experience", "in", "{SKILL}"],
    ["over", "{EXP}", "of", "professional", "experience"],
    ["total", "experience", "of", "{EXP}"],
    ["{EXP}", "experience", "in", "software", "development"],
    
    # Mixed
    ["worked", "at", "{ORG}", "for", "{EXP}", "using", "{SKILL}"],
    ["{EDU}", "graduate", "with", "{EXP}", "of", "experience"],
    ["developed", "{SKILL}", "solutions", "at", "{ORG}"],
    ["{EXP}", "at", "{ORG}", "working", "on", "{SKILL}", "projects"],
    ["from", "{DATE}", "to", "{DATE}", "worked", "at", "{ORG}"],
]


def _fill_entity(placeholder, word_to_ix):
    """Replace a placeholder with actual entity words and their BIO tags."""
    words = []
    tags = []
    
    if placeholder == "{SKILL}":
        if random.random() < 0.4 and MULTI_WORD_SKILLS:
            skill = random.choice(MULTI_WORD_SKILLS)
            for i, w in enumerate(skill):
                words.append(w)
                tags.append("B-SKILL" if i == 0 else "I-SKILL")
        else:
            w = random.choice(SKILLS_LIST)
            words.append(w)
            tags.append("B-SKILL")
    
    elif placeholder == "{ORG}":
        if random.random() < 0.3 and MULTI_WORD_ORGS:
            org = random.choice(MULTI_WORD_ORGS)
            for i, w in enumerate(org):
                words.append(w)
                tags.append("B-ORG" if i == 0 else "I-ORG")
        else:
            w = random.choice(ORGS)
            words.append(w)
            tags.append("B-ORG")
    
    elif placeholder == "{EDU}":
        w = random.choice(EDUCATION_TERMS)
        words.append(w)
        tags.append("B-EDU")
    
    elif placeholder == "{MULTI_EDU}":
        edu = random.choice(MULTI_WORD_EDU)
        for i, w in enumerate(edu):
            words.append(w)
            tags.append("B-EDU" if i == 0 else "I-EDU")
    
    elif placeholder == "{UNI}":
        if random.random() < 0.4 and MULTI_WORD_UNIS:
            uni = random.choice(MULTI_WORD_UNIS)
            for i, w in enumerate(uni):
                words.append(w)
                tags.append("B-ORG" if i == 0 else "I-ORG")
        else:
            w = random.choice(UNIVERSITIES)
            words.append(w)
            tags.append("B-ORG")
    
    elif placeholder == "{DATE}":
        date = random.choice(DATE_PATTERNS)
        for i, w in enumerate(date):
            words.append(w)
            tags.append("B-DATE" if i == 0 else "I-DATE")
    
    elif placeholder == "{EXP}":
        exp = random.choice(EXP_PATTERNS)
        for i, w in enumerate(exp):
            words.append(w)
            tags.append("B-EXP" if i == 0 else "I-EXP")
    
    return words, tags


def generate_ner_dataset(num_samples=2000, seed=42):
    """
    Generate synthetic BIO-tagged resume sentences for NER training.
    
    Returns:
        sentences: list of list of words
        tag_sequences: list of list of BIO tags
        word_to_ix: vocabulary mapping
    """
    random.seed(seed)
    
    sentences = []
    tag_sequences = []
    
    for _ in range(num_samples):
        template = random.choice(SENTENCE_TEMPLATES)
        sent_words = []
        sent_tags = []
        
        for token in template:
            if token.startswith("{"):
                words, tags = _fill_entity(token, None)
                sent_words.extend(words)
                sent_tags.extend(tags)
            else:
                sent_words.append(token)
                sent_tags.append("O")
        
        sentences.append(sent_words)
        tag_sequences.append(sent_tags)
    
    # Build vocabulary
    word_to_ix = {"<PAD>": 0, "<UNK>": 1}
    for sent in sentences:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    
    return sentences, tag_sequences, word_to_ix


def sentence_to_tensor(sentence, word_to_ix):
    """Convert word list to index tensor."""
    idxs = [word_to_ix.get(w, word_to_ix["<UNK>"]) for w in sentence]
    return torch.tensor(idxs, dtype=torch.long)


def tags_to_tensor(tags):
    """Convert tag list to index tensor."""
    idxs = [TAG_TO_IX[t] for t in tags]
    return torch.tensor(idxs, dtype=torch.long)


# ============================================================
# Training Function
# ============================================================
def train_ner_model(num_samples=2000, epochs=25, lr=0.01, save_path="ner_model.pth"):
    """
    Train the Bi-LSTM + CRF NER model on synthetic resume data.
    
    Args:
        num_samples: Number of training sentences to generate
        epochs: Training epochs
        lr: Learning rate
        save_path: Path to save trained model
    
    Returns:
        model: Trained BiLSTM_CRF model
        word_to_ix: Vocabulary mapping
    """
    print("=" * 60)
    print("  Training Bi-LSTM + CRF NER Model")
    print("=" * 60)
    
    # Generate data
    print(f"\n[1/3] Generating {num_samples} synthetic BIO-tagged sentences...")
    sentences, tag_sequences, word_to_ix = generate_ner_dataset(num_samples)
    print(f"      Vocabulary size: {len(word_to_ix)}")
    print(f"      Number of entity tags: {NUM_TAGS}")
    print(f"      Sample sentence: {' '.join(sentences[0])}")
    print(f"      Sample tags:     {' '.join(tag_sequences[0])}")
    
    # Split into train/validation (90/10)
    split = int(0.9 * len(sentences))
    train_sents, val_sents = sentences[:split], sentences[split:]
    train_tags, val_tags = tag_sequences[:split], tag_sequences[split:]
    
    # Initialize model
    print(f"\n[2/3] Initializing BiLSTM-CRF model...")
    model = BiLSTM_CRF(vocab_size=len(word_to_ix), embedding_dim=64, hidden_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"      Total parameters: {total_params:,}")
    print(f"      Architecture: Embedding(64) → BiLSTM(128) → Linear({NUM_TAGS}) → CRF")
    
    # Training loop
    print(f"\n[3/3] Training for {epochs} epochs...")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        random.shuffle(list(zip(train_sents, train_tags)))
        
        for sent, tags in zip(train_sents, train_tags):
            model.zero_grad()
            
            sent_tensor = sentence_to_tensor(sent, word_to_ix)
            tags_tensor = tags_to_tensor(tags)
            
            loss = model.neg_log_likelihood(sent_tensor, tags_tensor)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients in LSTM
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / len(train_sents)
        
        # Validation accuracy
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for sent, tags in zip(val_sents, val_tags):
                    sent_tensor = sentence_to_tensor(sent, word_to_ix)
                    _, pred_tags = model(sent_tensor)
                    gold_tags = [TAG_TO_IX[t] for t in tags]
                    for p, g in zip(pred_tags, gold_tags):
                        total += 1
                        if p == g:
                            correct += 1
            val_acc = correct / total * 100 if total > 0 else 0
            print(f"      Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.1f}%")
            model.train()
    
    # Save model and vocabulary
    model.eval()
    save_data = {
        "model_state_dict": model.state_dict(),
        "word_to_ix": word_to_ix,
        "vocab_size": len(word_to_ix),
        "embedding_dim": 64,
        "hidden_dim": 128,
    }
    torch.save(save_data, save_path)
    print(f"\n      Model saved to {save_path}")
    print("=" * 60)
    
    return model, word_to_ix


# ============================================================
# Inference: Extract Entities from Raw Text
# ============================================================
def load_ner_model(model_path="ner_model.pth"):
    """Load a trained BiLSTM-CRF model from disk."""
    data = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    model = BiLSTM_CRF(
        vocab_size=data["vocab_size"],
        embedding_dim=data["embedding_dim"],
        hidden_dim=data["hidden_dim"]
    )
    model.load_state_dict(data["model_state_dict"])
    model.eval()
    return model, data["word_to_ix"]


def extract_entities(text, model, word_to_ix):
    """
    Extract named entities from raw text using the trained BiLSTM-CRF model.
    
    Args:
        text: Raw text string (e.g., resume content)
        model: Trained BiLSTM_CRF model
        word_to_ix: Vocabulary mapping
    
    Returns:
        Dictionary with extracted entities grouped by type:
        {
            "skills": ["python", "machine learning", ...],
            "organizations": ["google", "microsoft", ...],
            "education": ["bachelor", "computer science", ...],
            "dates": ["2020", "january 2021", ...],
            "experience": ["3 years", "6 months", ...]
        }
    """
    # Tokenize
    words = text.lower().split()
    if not words:
        return {"skills": [], "organizations": [], "education": [], "dates": [], "experience": []}
    
    # Convert to tensor
    word_indices = [word_to_ix.get(w, word_to_ix.get("<UNK>", 1)) for w in words]
    sent_tensor = torch.tensor(word_indices, dtype=torch.long)
    
    # Predict
    with torch.no_grad():
        _, pred_tag_ids = model(sent_tensor)
    
    pred_tags = [IX_TO_TAG.get(t, "O") for t in pred_tag_ids]
    
    # Group entities by BIO tags
    entities = {"skills": [], "organizations": [], "education": [], "dates": [], "experience": []}
    
    tag_to_entity_key = {
        "SKILL": "skills",
        "ORG": "organizations",
        "EDU": "education",
        "DATE": "dates",
        "EXP": "experience"
    }
    
    current_entity = []
    current_type = None
    
    for word, tag in zip(words, pred_tags):
        if tag.startswith("B-"):
            # Save previous entity if exists
            if current_entity and current_type:
                key = tag_to_entity_key.get(current_type)
                if key:
                    entities[key].append(" ".join(current_entity))
            # Start new entity
            current_type = tag[2:]
            current_entity = [word]
        elif tag.startswith("I-") and current_type == tag[2:]:
            # Continue current entity
            current_entity.append(word)
        else:
            # Save previous entity if exists
            if current_entity and current_type:
                key = tag_to_entity_key.get(current_type)
                if key:
                    entities[key].append(" ".join(current_entity))
            current_entity = []
            current_type = None
    
    # Don't forget the last entity
    if current_entity and current_type:
        key = tag_to_entity_key.get(current_type)
        if key:
            entities[key].append(" ".join(current_entity))
    
    # Deduplicate
    for key in entities:
        entities[key] = list(set(entities[key]))
    
    return entities


# ============================================================
# Main (standalone test)
# ============================================================
if __name__ == "__main__":
    model, word_to_ix = train_ner_model(num_samples=2000, epochs=25, save_path="ner_model.pth")
    
    # Test inference
    test_text = "experienced in python and machine learning worked at google for 3 years bachelor in computer science"
    entities = extract_entities(test_text, model, word_to_ix)
    print(f"\nTest text: {test_text}")
    print(f"Extracted entities: {json.dumps(entities, indent=2)}")
