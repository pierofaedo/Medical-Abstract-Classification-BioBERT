# ==============================================================================
# 02_01: ZERO-SHOT LEARNING BASELINE (BART-LARGE-MNLI)
# ==============================================================================
# Objective: Establish a high-performance Transformer baseline quickly.
# Model: facebook/bart-large-mnli
# 
# TECHNICAL INSIGHT & STRATEGIC DECISION:
# 1. SEMANTIC REASONING: Unlike BioBERT, BART uses "Semantic Reasoning". It maps 
#    the abstract to natural language labels (e.g., "neoplasms") by understanding 
#    the linguistic relationship between the text and the label name itself. 
#    It doesn't need to see the training data to make an educated guess.
#
# 2. EFFICIENCY: While BART can be fine-tuned, a Zero-Shot approach was 
#    intentionally selected here to bypass high local computational costs 
#    and training time, while still achieving a significant accuracy boost.
#
# Accuracy achieved: 0.552 (Significant boost over 0.434 R baseline)
# ==============================================================================

import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm
import os

# 1. Data Loading
file_path = "medical_abstracts_validation.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Missing {file_path}. Please place it in the script directory.")

df = pd.read_csv(file_path)

# 2. Hardware Acceleration Check
device = 0 if torch.cuda.is_available() else -1
print(f"\nSYSTEM READY. EXECUTION UNIT: {'NVIDIA GPU' if device == 0 else 'CPU'}")

# 3. Model Initialization (Zero-Shot Pipeline)
# This model uses Natural Language Inference (NLI) to classify text 
# by checking if the abstract logically entails the category label.
print("Loading BART Model from Hugging Face... (Large architecture)")
classifier = pipeline("zero-shot-classification", 
                      model="facebook/bart-large-mnli", 
                      device=device)

# 4. Semantic Category Mapping
# BART "reasons" by comparing the abstract to the meaning of these specific words.
class_mapping = {
    "digestive system diseases": 1,
    "cardiovascular diseases": 2,
    "neoplasms": 3,
    "nervous system diseases": 4,
    "general medicine": 5
}
labels = list(class_mapping.keys())

# --- PHASE 1: SEMANTIC PREVIEW ---
print("\n--- INITIAL INFERENCE PREVIEW (Semantic Reasoning) ---")
for i in range(2):
    text = df['medical_abstract'].iloc[i]
    res = classifier(text, candidate_labels=labels)
    predicted_label = res['labels'][0]
    confidence = res['scores'][0]
    print(f"Abstract {i+1}: Predicted as '{predicted_label}' ({confidence:.2%} confidence)")
    print(f"Mapped to Code: {class_mapping[predicted_label]}\n")

# --- PHASE 2: FULL BATCH PROCESSING ---
final_predictions = []

print("Processing 2,000 abstracts... Running Inference.")
for text in tqdm(df['medical_abstract']):
    res = classifier(text, candidate_labels=labels)
    top_label = res['labels'][0]
    code = class_mapping[top_label]
    final_predictions.append(code)

# 5. Export Submission File
output_file = "predictions_transformer_zeroshot.txt"
with open(output_file, "w") as f:
    for pred in final_predictions:
        f.write(f"{pred}\n")

print(f"\n\nCOMPLETED! File '{output_file}' created successfully.")
print(f"Total processed rows: {len(final_predictions)}")
