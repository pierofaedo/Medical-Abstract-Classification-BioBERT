# Medical Abstract Classification: From Statistical Baselines to BioBERT 🧬

This repository contains the full research workflow and winning solution for a **Medical NLP Challenge** (Master’s Degree in Statistical Sciences, University of Padua). 

I achieved **1st Place** on the final leaderboard with an accuracy of **0.652**, utilizing domain-specific Transformer architectures and high-performance computing.

---

## 📌 Project Overview
The goal of the challenge was to classify **2,000 medical abstracts** into 5 distinct categories, using a provided training set of **9,000 labeled observations**. 

The project followed an iterative research workflow, moving from robust ensemble-based baselines to specialized Deep Learning models. Each phase was validated through a **Proof of Concept (PoC)** to ensure pipeline stability before scaling to full-dataset training.

---

## 📈 Performance Evolution
| Phase | Model | Tools | Training Samples | Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **01** | **Random Forest** | R (Local) | 9,000 | 0.434 |
| **02** | **BART Zero-Shot** | Python (Local) | 0 (Zero-Shot) | 0.552 |
| **03** | **BioBERT PoC** | Python (Local) | 100 | 0.340* |
| **04** | **BioBERT Final** | **Python (Colab GPU)** | **9,000** | **0.652** |

*\*The PoC (Phase 03) was used solely to validate the technical pipeline (tokenization, CUDA, feature extraction) before scaling to the full dataset.*

---

## 🧪 Methodology & Model Evolution
The research followed a logical progression to identify the most efficient architecture:

1. **Statistical Baseline (Random Forest):** Established a performance floor using TF-IDF and Bag-of-Words. While robust, it failed to capture the deep clinical context of the abstracts.
2. **Semantic Reasoning (BART Zero-Shot):** Implemented `facebook/bart-large-mnli` to test if understanding label names (e.g., "neoplasms") improved performance. The jump to 0.552 confirmed that semantic context is superior to simple word frequency.
3. **Domain Adaptation (BioBERT):** Final fine-tuning of `dmis-lab/biobert-v1.1` (pre-trained on PubMed/PMC). This allowed the model to recognize specialized clinical patterns that general-purpose models miss.

---

## 📉 Training Logs & Technical Specs
The final model was trained on **Google Colab (Tesla T4 GPU)** using **FP16 Mixed Precision** to optimize memory and speed.

**Final Training Convergence:**
```text
Step | Training Loss
-----+--------------
100  | 1.3263
500  | 0.8400
1000 | 0.7147
1500 | 0.6084
1791 | 0.5892 (Final)
```
The steady decline in loss from 1.32 to 0.58 confirmed proper weight optimization during the 3 epochs of fine-tuning.

---

## 🛠️ Advanced Feature Extraction
While the final feature CSV is not uploaded due to size constraints, the pipeline is designed to extract:

* **Contextual Embeddings:** 768-dimensional vectors from the last hidden layer (CLS token) for semantic similarity analysis.
* **Softmax Probabilities:** Class-specific confidence scores for downstream error analysis or potential model stacking.

---

## 📂 Project Structure
* **`01_r_baseline/`**: Initial exploration and Random Forest scripts (R).
    * `01_01_random_forest_analysis.R`
* **`02_transformers_exploration/`**:
    * `02_01_zero_shot_bart.py`: Semantic reasoning test.
    * `02_02_biobert_pipeline_validation.py`: Technical PoC on 100 samples.
* **`03_biobert_final_model_and_results/`**:
    * `03_01_biobert_full_finetuning.py`: Winning model script.
    * `03_02_final_predictions.txt`: Submission file for the 2,000 validation samples.

---

## 👤 Author
**Piero Faedo** 📧 [piero.faedo@outlook.com](mailto:piero.faedo@outlook.com)  
