# Medical Abstract Classification: From Statistical Baselines to BioBERT 🏆

This repository contains my winning solution for the Medical NLP Challenge, 
where I achieved **1st Place** with an accuracy of **0.652**.

## 📌 Project Overview
The goal was to classify 9,000+ medical abstracts into 5 distinct categories. 
I followed an iterative research approach, moving from traditional machine learning 
to state-of-the-art Transformers.

## 📈 Performance Evolution
| Phase | Model | Tools | Accuracy |
| :--- | :--- | :--- | :--- |
| **01** | **Random Forest** | R (Local) | 0.434 |
| **02** | **BERT Zero-Shot** | R (Local) | 0.552 |
| **03** | **BioBERT Fine-Tuning** | **Python (Colab GPU)** | **0.652** |

## 🛠️ Key Technical Decisions
- **BioBERT Integration:** Used `dmis-lab/biobert-v1.1` for its superior understanding of clinical terminology.
- **Cloud Computing:** Leveraged Google Colab (Tesla T4 GPU) to reduce training time from 15 hours to 45 minutes.
- **Robust Pipeline:** Extracted 768-dimensional embeddings and class probabilities for downstream analysis in R.

## 📂 Contents
- `BioBERT_Training.ipynb`: Full fine-tuning and inference notebook.
- `R_scripts/`: Initial R explorations and baseline models.
- `previsions/`: Final classification output.

---
**Piero Faedo** [piero.faedo@outlook.com](mailto:piero.faedo@outlook.com)
