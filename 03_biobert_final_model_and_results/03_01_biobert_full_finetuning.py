# ==============================================================================
# 03_01: FINAL BIOBERT FINE-TUNING (FULL DATASET - 9,000 SAMPLES)
# ==============================================================================
# Objective: Maximize clinical abstract classification via supervised learning.
# Model: dmis-lab/biobert-v1.1 (Pre-trained on PubMed/PMC)
# 
# TECHNICAL STRATEGY:
# 1. HARDWARE OPTIMIZATION: Used FP16 (Mixed Precision) to accelerate training 
#    on NVIDIA Tesla T4 GPU, reducing training time to ~45-50 minutes.
# 2. FEATURE EXTRACTION: Beyond classification, the script extracts the 
#    768-dimensional CLS embeddings for potential stacking or clustering.
# 3. DOMAIN ADAPTATION: Unlike general BART, BioBERT identifies specialized 
#    medical tokens (e.g., specific pathology terminology) mapping them to codes.
#
# Accuracy achieved: 0.652
# ==============================================================================

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import numpy as np
import os

# 1. DATA LOADING
# Note: Paths updated for repository consistency.
path_train = "medical_abstracts_train.csv"
path_test = "medical_abstracts_validation.csv"

train_df = pd.read_csv(path_train)
test_df = pd.read_csv(path_test)

# Map labels 1-5 to 0-4 for CrossEntropyLoss compatibility
train_df['label'] = train_df['condition_label'] - 1

# 2. BIOBERT CONFIGURATION
model_name = "dmis-lab/biobert-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

class MedicalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self): return len(self.texts)
    
    def __getitem__(self, item):
        encoding = self.tokenizer(
            str(self.texts[item]), 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_len, 
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[item], dtype=torch.long)
        }

train_dataset = MedicalDataset(train_df['medical_abstract'].values, train_df['label'].values, tokenizer)

# 3. FINAL TRAINING SETUP
# Initializing the model with 5 output heads
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5).to("cuda")

training_args = TrainingArguments(
    output_dir='./results_final',
    num_train_epochs=3,              # 3 complete passes over 9,000 records
    per_device_train_batch_size=16,  # Optimized for 16GB VRAM (Tesla T4)
    learning_rate=2e-5,
    logging_steps=100,
    save_strategy="no",
    fp16=True,                       # Mixed Precision for speed/memory efficiency
    report_to="none"
)

trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)

print("--- STARTING FINAL TRAINING (Approx. 45-50 min) ---")
trainer.train()

# 4. INFERENCE & FEATURE EXTRACTION (2,000 Validation Records)
model.eval()
predictions_list = []
all_probs = []
all_embeddings = []

print("---  EXTRACTING PREDICTIONS AND EMBEDDINGS ---")
with torch.no_grad():
    for text in test_df['medical_abstract']:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=512).to("cuda")
        outputs = model(**inputs, output_hidden_states=True)

        # A. Classification (Converting back to 1-5 scale)
        pred = torch.argmax(outputs.logits, dim=-1).item()
        predictions_list.append(pred + 1)

        # B. Softmax Probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        all_probs.append(probs)

        # C. Contextual Embeddings (Last Hidden Layer, CLS token)
        emb = outputs.hidden_states[-1][:, 0, :].cpu().numpy()[0]
        all_embeddings.append(emb)

# 5. DATA EXPORT
# Final submission file
with open("final_predictions_biobert.txt", "w") as f:
    for p in predictions_list:
        f.write(f"{p}\n")

# Comprehensive CSV for Downstream Stacking or R Analysis
df_probs = pd.DataFrame(all_probs, columns=[f"prob_cl_{i+1}" for i in range(5)])
df_embs = pd.DataFrame(all_embeddings, columns=[f"emb_{i}" for i in range(768)])
final_output = pd.concat([test_df[['medical_abstract']].reset_index(drop=True), df_probs, df_embs], axis=1)
final_output.to_csv("biobert_final_features.csv", index=False)

print("Process completed. Output files generated.")
print("Files 'final_predictions_biobert.txt' and 'biobert_final_features.csv' generated.")
