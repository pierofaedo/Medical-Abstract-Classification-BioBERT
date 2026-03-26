# Medical Abstract Classification: From Ensemble Learning (Random Forest) to Deep Learning (BioBERT) 🏆

This repository contains my winning solution for a **Medical NLP Challenge** held within the **Master's Degree in Statistical Sciences** at the **University of Padua**.

I achieved **1st Place** in the final leaderboard with an accuracy of **0.652**, leveraging domain-specific Transformer architectures.

## 📌 Project Overview
The goal was to classify **2,000 medical abstracts** into 5 distinct categories, using a provided training set of **9,000 labeled observations**. 
The project followed an iterative research workflow, moving from robust ensemble-based baselines to specialized Deep Learning models.

## 📈 Performance Evolution
| Phase | Model | Tools | Accuracy |
| :--- | :--- | :--- | :--- |
| **01** | **Random Forest** (Ensemble) | R (Local) | 0.434 |
| **02** | **BERT Zero-Shot** | R (Local) | 0.552 |
| **03** | **BioBERT Fine-Tuning** | **Python (Colab GPU)** | **0.652** |

## 🛠️ Key Technical Decisions
- **Domain-Specific Transformers:** Utilized `dmis-lab/biobert-v1.1` (pre-trained on PubMed/PMC) to capture complex clinical terminology far more effectively than general-purpose models.
- **Compute Strategy:** Migrated the pipeline to Google Colab (Tesla T4 GPU) to handle the computational intensity of Transformer fine-tuning, optimizing training time from hours to minutes.
- **Methodology:** Established a performance floor using a **Random Forest** baseline before implementing a full fine-tuning pipeline for the final 2,000-sample validation set.

## 📂 Contents
- `BioBERT_Training.ipynb`: Full fine-tuning and inference notebook (Python).
- `R_scripts/`: Initial exploration and Random Forest/BERT baseline scripts (R).
- `previsions/`: Final classification output for the 2,000 validation samples.

---
---
**Piero Faedo** 📧 [piero.faedo@outlook.com](mailto:piero.faedo@outlook.com)
