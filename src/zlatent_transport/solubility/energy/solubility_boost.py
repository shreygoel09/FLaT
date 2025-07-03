#!/usr/bin/env python3

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import pearsonr
from xgboost import XGBClassifier
from omegaconf import OmegaConf
from src.utils.model_utils import _print

config = OmegaConf.load("/home/a03-sgoel/FLaT/src/configs/solubility.yaml")

# Load ESM-2 tokenizer and model (no gradients needed)
tokenizer = AutoTokenizer.from_pretrained(config.lm.pretrained_esm)
model = AutoModel.from_pretrained(config.lm.pretrained_esm).eval().cuda()

@torch.no_grad()
def get_mean_embedding(sequence):
    tokens = tokenizer(sequence, return_tensors="pt", truncation=True, padding=True)
    input_ids = tokens["input_ids"].cuda()
    attention_mask = tokens["attention_mask"].cuda()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    embeddings = outputs.last_hidden_state  # [B, L, D]
    mean_embeds = (embeddings * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
    return mean_embeds.squeeze(0).cpu().numpy()

def embed_dataset(df):
    X, y = [], []
    for seq, label in tqdm(zip(df['Sequence'], df['Label']), total=len(df)):
        X.append(get_mean_embedding(seq))
        y.append(label)
    return np.array(X), np.array(y)

# Load data
train_df = pd.read_csv(config.data.train)
val_df = pd.read_csv(config.data.val)
test_df = pd.read_csv(config.data.test)

# Combine val + test for final evaluation
combined_test_df = pd.concat([val_df, test_df], ignore_index=True)

# Get embeddings
X_train, y_train = embed_dataset(train_df)
X_test, y_test = embed_dataset(combined_test_df)

# Train XGBoost
clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
clf.fit(X_train, y_train)

# Predict
y_pred_prob = clf.predict_proba(X_test)[:, 1]
y_pred = (y_pred_prob >= 0.5).astype(int)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
auroc = roc_auc_score(y_test, y_pred_prob)
pearson = pearsonr(y_test, y_pred_prob)[0]

_print(f"\nTest Set Results:")
_print(f"Accuracy: {accuracy:.4f}")
_print(f"AUROC:    {auroc:.4f}")
_print(f"Pearson:  {pearson:.4f}")
