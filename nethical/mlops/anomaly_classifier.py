# Single Cell - Full Transformer-Based Anomaly Detection Classifier

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter
from typing import List, Dict, Any, Tuple
from sklearn.metrics import roc_auc_score
from scipy.stats import entropy as scipy_entropy

# --- Data Utilities ---

def build_tokenizer(sequences):
    """Creates a dictionary mapping actions to integer IDs."""
    actions = set(a for seq in sequences for a in seq)
    tok = {a: i+1 for i, a in enumerate(sorted(actions))}  # 0 is reserved for padding
    return tok

def encode_sequences(sequences, tok, max_len=None):
    """Encodes action sequences using tokenizer."""
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    encoded = []
    for seq in sequences:
        seq_ids = [tok[a] for a in seq]
        if len(seq_ids) < max_len:
            seq_ids += [0]*(max_len - len(seq_ids))
        else:
            seq_ids = seq_ids[:max_len]
        encoded.append(seq_ids)
    return np.array(encoded)

def inverse_tokenizer(tok):
    return {v:k for k,v in tok.items()}

# --- Feature Engineering ---

def extract_features(sequence):
    features = {}

    # N-gram rarity
    n = 3
    ngrams = [tuple(sequence[i:i+n]) for i in range(len(sequence)-n+1)]
    features['ngram_count'] = len(ngrams)
    features['unique_ngrams'] = len(set(ngrams))/max(len(ngrams),1)
    
    # Action frequency deviation
    act_counts = Counter(sequence)
    freq = [v/len(sequence) for v in act_counts.values()]
    features['freq_entropy'] = scipy_entropy(freq)
    
    # Sequence entropy (Shannon)
    features['shannon_entropy'] = scipy_entropy([act_counts[a]/len(sequence) for a in act_counts])
    
    # Pattern diversity
    features['pattern_diversity'] = len(set(sequence))/len(sequence)
    
    # Markov transitions
    transitions = [tuple(sequence[i:i+2]) for i in range(len(sequence)-1)]
    trans_counts = Counter(transitions)
    trans_freq = [v/len(transitions) for v in trans_counts.values()]
    features['markov_entropy'] = scipy_entropy(trans_freq)
    
    # Palindromic pattern detection
    features['is_palindrome'] = int(sequence == sequence[::-1])
    
    # Rare event statistics
    rare_events = [action for action, cnt in act_counts.items() if cnt < (0.05 * len(sequence))]
    features['rare_event_ratio'] = len(rare_events)/len(set(sequence))
    
    # Interleaved actions (count alternations)
    alternations = sum(1 for i in range(1,len(sequence)) if sequence[i] != sequence[i-1])
    features['alternation_ratio'] = alternations/len(sequence)
    
    # Motif detection: looks for certain subsequences (here, all sequences of length 2)
    unique_pairs = set(tuple(sequence[i:i+2]) for i in range(len(sequence)-1))
    features['motif_count'] = len(unique_pairs)
    
    # Duration/frequency statistics
    features['avg_action_runlength'] = np.mean([len(list(group)) for _, group in groupby(sequence)])
    
    # Position-based (are anomalies at head/tail)
    features['head_action'] = sequence[0]
    features['tail_action'] = sequence[-1]
    
    return features

# --- Transformer Model ---

class TransformerAnomalyClassifier(nn.Module):
    def __init__(self, vocab_size, feature_dim, d_model=64, nhead=4, num_layers=2, max_len=48):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size+1, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Feature input
        self.feature_fc = nn.Linear(feature_dim, d_model)
        # Output
        self.final_fc = nn.Linear(d_model*2, 1)
        
    def forward(self, x, features):
        # x: [B, L]
        B, L = x.shape
        x_emb = self.embedding(x)                   # [B, L, d_model]
        pos_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        pos_emb = self.pos_embedding(pos_ids)       # [B, L, d_model]
        h_seq = self.transformer(x_emb + pos_emb)   # [B, L, d_model]
        h_seq_sum = h_seq.mean(dim=1)               # [B, d_model] (simple pooling)
        
        h_feat = self.feature_fc(features)          # [B, d_model]
        h = torch.cat([h_seq_sum, h_feat], dim=1)   # [B, 2*d_model]
        out = self.final_fc(h)
        return torch.sigmoid(out.squeeze(1))        # [B]

# --- Demo Training and Prediction ---

# Simulated synthetic data:
samples = [
    {'sequence': ['login','view','logout','login','download','logout'], 'label':0},
    {'sequence': ['login','download','download','logout','login','download'], 'label':1},
    {'sequence': ['login','view','view','logout','logout','view'], 'label':0},
    {'sequence': ['login','download','delete','login','delete','logout'], 'label':1},
    {'sequence': ['login','view','download','view','logout','logout'], 'label':0},
    {'sequence': ['download']*6, 'label':1},
    {'sequence': ['login','view','view','login','view','logout'], 'label':0},
    {'sequence': ['download','delete','download','delete','download','delete'], 'label':1},
]

sequences = [s['sequence'] for s in samples]
labels = torch.tensor([s['label'] for s in samples], dtype=torch.float32)

# Build tokenizer and encode
tok = build_tokenizer(sequences)
max_len = max(len(seq) for seq in sequences)
X = encode_sequences(sequences, tok, max_len)
inv_tok = inverse_tokenizer(tok)

# Extract engineered features
import pandas as pd
from itertools import groupby

feat_dicts = [extract_features(seq) for seq in sequences]
# Convert categorical (head/tail actions) to int features:
for fd in feat_dicts:
    fd['head_action'] = tok.get(fd['head_action'],0)
    fd['tail_action'] = tok.get(fd['tail_action'],0)

df_feat = pd.DataFrame(feat_dicts).fillna(0)
features = torch.tensor(df_feat.values, dtype=torch.float32)
feature_dim = features.shape[1]

# Instantiate model
model = TransformerAnomalyClassifier(
    vocab_size=len(tok),
    feature_dim=feature_dim,
    d_model=64,
    nhead=4,
    num_layers=2,
    max_len=max_len
)

# --- Training loop ---
epochs = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
X_tensor = torch.tensor(X, dtype=torch.long).to(device)
features_tensor = features.to(device)
labels_tensor = labels.to(device)

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    preds = model(X_tensor, features_tensor)
    loss = nn.BCELoss()(preds, labels_tensor)
    loss.backward()
    optimizer.step()
    if (epoch+1)%5==0:
        print(f"Epoch {epoch+1}: Loss={loss.item():.4f}")

# --- Prediction example ---
model.eval()
with torch.no_grad():
    # Predict label (0=normal, 1=anomalous) for all samples
    logits = model(X_tensor, features_tensor)
    print("Predictions:", (logits.cpu().numpy() > 0.5).astype(int))
    try:
        print("AUC:", roc_auc_score(labels.cpu(), logits.cpu()))
    except Exception:
        pass
    
# --- Predict on new sequence ---
new_seq = ['login','download','delete','logout','download','logout']
new_x = encode_sequences([new_seq], tok, max_len)
new_feat_dict = extract_features(new_seq)
new_feat_dict['head_action'] = tok.get(new_feat_dict['head_action'],0)
new_feat_dict['tail_action'] = tok.get(new_feat_dict['tail_action'],0)
new_features = torch.tensor([[new_feat_dict[k] for k in df_feat.columns]], dtype=torch.float32)

with torch.no_grad():
    out = model(torch.tensor(new_x, dtype=torch.long).to(device), new_features.to(device))
    print("New sequence anomaly score:", out.item())
    print("Label (1=anomalous):", int(out.item() > 0.5))
