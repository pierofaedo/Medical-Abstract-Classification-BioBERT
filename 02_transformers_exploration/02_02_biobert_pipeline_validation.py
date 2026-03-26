# ==============================================================================
# 02_02: BIOBERT PIPELINE VALIDATION (TECHNICAL PROOF OF CONCEPT)
# ==============================================================================
# Objective: Verify the end-to-end training and inference pipeline locally.
# Model: dmis-lab/biobert-v1.1 (Specialized for Biomedical text)
# 
# TECHNICAL INSIGHT:
# Unlike BART (Zero-Shot), BioBERT requires Supervised Fine-Tuning. It doesn't 
# "reason" about label names; it learns statistical patterns between medical 
# tokens and numeric classes. 
#
# PURPOSE OF THIS SCRIPT:
# This is a "Sanity Check" performed on a tiny subset (10 samples) to ensure:
# 1. CUDA/GPU compatibility and memory management.
# 2. Correct extraction of CLS embeddings (768-dimensional vectors).
# 3. Proper Softmax probability distribution across the 5 medical classes.
# ==============================================================================

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import numpy as np
import time
import os

# 1. DATA LOADING (Subsampling for rapid pipeline testing)
# Using a 10-sample slice to validate the logic without heavy compute.
train_path = "medical_abstracts_train.csv"
test_path = "medical_abstracts_validation.csv"

# Safety check for local files
if not os.path.exists(train_path) or not os.path.exists(test_path):
    print("Note: Data files not found. Ensure CSVs are in the working directory.")
    # For GitHub demonstration, we assume files are present or handled via git-lfs
else:
    train_df_full = pd.read_csv(train_path)
    test_df_full = pd.read_csv(test_path)

    train_df = train_df_full.head(10).copy()
    test_df = test_df_full.head(10).copy()

    # Align labels: mapping 1-5 to 0-4 for CrossEntropyLoss compatibility
    train_df['label'] = train_df['condition_label'] - 1

# 2. BIOBERT CONFIGURATION
# Model weights specialized on PubMed and PMC corpora.
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

# 3. MODEL SETUP & HARDWARE CHECK
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"EXECUTION DEVICE: {device.upper()}")

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5).to(device)

# Minimal arguments for pipeline validation
training_args = TrainingArguments(
    output_dir='./results_test',
    num_train_epochs=1,
    per_device_train_batch_size=2,
    logging_steps=1,
    save_strategy="no",
    use_cpu=(device == "cpu"),
    report_to="none" # Disabling external logging for the test
)

trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)

print("\n--- STARTING TECHNICAL TEST: 10 RECORDS TRAINING ---")
start_time = time.time()
trainer.train()
end_time = time.time()
print(f"Training Time (10 samples): {round(end_time - start_time, 2)} seconds")

# 4. FEATURE EXTRACTION (PROBABILITIES & EMBEDDINGS)
# This phase tests the extraction of "hidden" model features for downstream analysis.
print("\n--- INFERENCE TEST: EXTRACTING FEATURES ---")
model.eval()
all_probs = []
all_embeddings = []

with torch.no_grad():
    for text in test_df['medical_abstract']:
        # Prepare inputs
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=512).to(device)
        
        # Explicitly request hidden_states for embedding extraction
        outputs = model(**inputs, output_hidden_states=True)
        
        # A. Class Probabilities (Softmax Layer)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        all_probs.append(probs)
        
        # B. Document Embeddings (Last hidden layer, CLS token at index 0)
        # Dimensions: 768 (Standard for BERT-base architectures)
        emb = outputs.hidden_states[-1][:, 0, :].cpu().numpy()[0]
        all_embeddings.append(emb)

# 5. DATA EXPORT FOR VERIFICATION
df_probs = pd.DataFrame(all_probs, columns=[f"prob_cl_{i+1}" for i in range(5)])
df_embs = pd.DataFrame(all_embeddings, columns=[f"emb_{i}" for i in range(768)])

final_test_output = pd.concat([test_df[['medical_abstract']].reset_index(drop=True), df_probs, df_embs], axis=1)
final_test_output.to_csv("test_biobert_output.csv", index=False)

print("\n✅ PIPELINE VALIDATION COMPLETED!")
print(f"Generated 'test_biobert_output.csv' with {final_test_output.shape[1]} features.")
