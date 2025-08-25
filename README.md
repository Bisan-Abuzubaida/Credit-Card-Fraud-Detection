# Credit Card Fraud Detection — Hybrid Sampling + Random Forest (Outperforms Kaggle Baselines)

This project tackles the highly imbalanced **[Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)** using a **hybrid resampling approach (SMOTE + Random Undersampling)** combined with a tuned **Random Forest classifier**.  

Compared to the best-rated Kaggle baselines (SMOTE-only and Undersampling-only with Neural Nets), our approach achieves **superior balance across accuracy, recall, precision, and F1-score**, making it both **practical and reliable** for fraud detection.

---

## 📦 Project Overview

- **Dataset:** Kaggle “Credit Card Fraud Detection” (284,807 transactions, 492 frauds ≈ 0.17%)   
- **Challenge:** Extreme class imbalance  
- **Solution:**  
  1. **Cleaning & preprocessing**  
     - Verified no missing values  
     - Removed outliers only from the **majority (legitimate)** class
     - Dropped 11 weakly correlated features (`Time, V8, V13, V19, V22–V28`) → reduced features from 31 → **20 (plus Amount)**  
     - Min–Max scaling for consistent feature magnitudes  
  2. **Resampling strategy:** **Hybrid Sampling = SMOTE (oversampling) + Random Undersampling**  
  3. **Model:** **Random Forest (100 trees, tuned)**  
  4. **Evaluation:** Confusion matrix, accuracy, recall, precision, F1-score  

---

## ⚙️ Best Model: Random Forest + Hybrid Sampling

**Pipeline steps:**
1. Train/validation/test split  
2. Min–Max scaling  
3. Apply **SMOTE (sampling_strategy=0.5)**  
4. Apply **Random Undersampling (sampling_strategy=0.8)**  
5. Fit **RandomForestClassifier(n_estimators=100, random_state=42)**  
6. Save model with `joblib`  

---

## 📊 Results

### 🔹 My Hybrid Sampling (Random Forest)
- **Confusion Matrix:**  
  TN = 56,851 | FP = 13 | FN = 14 | TP = 84  

- **Metrics:**  
  - Accuracy: **99.95%**  
  - Precision (Fraud): **0.866**  
  - Recall (Fraud): **0.857**  
  - F1 (Fraud): **0.861**

---

### 🔹 Kaggle Baseline — SMOTE (Neural Net)
- **Confusion Matrix:**  
  TN = 56,851 | FP = 12 | FN = 33 | TP = 65  

- **Metrics:**  
  - Accuracy: **99.92%**  
  - Precision (Fraud): **0.844**  
  - Recall (Fraud): **0.663**  
  - F1 (Fraud): **0.742**

---

### 🔹 Kaggle Baseline — Undersampling (Neural Net)
- **Confusion Matrix:**  
  TN = 55,148 | FP = 1,715 | FN = 8 | TP = 90  

- **Metrics:**  
  - Accuracy: **97.03%**  
  - Precision (Fraud): **0.050**  
  - Recall (Fraud): **0.918**  
  - F1 (Fraud): **0.095**

---

## 🥇 Comparison — Hybrid vs Kaggle Baselines

| Method                     | Accuracy | Precision (Fraud) | Recall (Fraud) | F1 (Fraud) |
|-----------------------------|----------|-------------------|----------------|------------|
| **My Hybrid RF (SMOTE+RUS)** | **99.95%** | **0.866**          | **0.857**       | **0.861**   |
| Kaggle SMOTE (NN)          | 99.92%   | 0.844             | 0.663          | 0.742      |
| Kaggle Undersampling (NN)  | 97.03%   | 0.050             | **0.918**       | 0.095      |
 
**Key improvements of our Hybrid RF:**
- vs **Kaggle SMOTE:**  
  + **+19.4 percentage points recall** (0.857 vs 0.663)  
  + Higher F1 (+11.9 points)  
  + Slightly better accuracy (+0.03 pp)  

- vs **Kaggle Undersampling:**  
  + **+81.6 percentage points precision** (0.866 vs 0.050)  
  + F1 improved by **+76.6 points**  
  + Accuracy higher by **+2.92 pp**  
  + Recall slightly lower (0.857 vs 0.918), but balanced by massive gains in precision  

---

## 🖼️ Confusion Matrices (Visual)

### Our Hybrid Sampling (RF)
<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/44c23840-896c-4ce3-a17c-3644d61e9da0" />


### Kaggle SMOTE Baseline
<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/55130397-cf65-4daa-88eb-7ab2677524af" />

### Kaggle Undersampling Baseline
<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/08aa3f1d-6753-40a6-94a1-93c743f632a3" />

---

## 🧪 Reproduce Locally

1. Clone repo & install:

    ```bash
   git clone https://github.com/Bisan-Abuzubaida/Credit-Card-Fraud-Detection.git
   cd Credit-Card-Fraud-Detection
   python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Place creditcard.csv in project root.
3. Run notebook:
```
jupyter notebook "Credit Card Fraud Detection.ipynb"
```
4. Artifacts:

  - Best model → `random_forest_model.joblib`

  - Confusion matrices & plots
    
---

📁 Repository Structure
<pre>
├── Credit Card Fraud Detection.ipynb      # Full workflow
├── random_forest_model.joblib             # Saved best model
├── requirements.txt
└── README.md
</pre>

---

🚀 Why This Project Stands Out

•	Hybrid Sampling: Outperforms single SMOTE or undersampling approaches.

•	Balanced Performance: High accuracy and strong fraud recall with realistic precision.

•	Feature Engineering: Reduced 31 → 20 features without sacrificing signal.

•	Production Ready: Clean notebook pipeline + saved model artifact.

---

🙌 Acknowledgments

•	Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

•	Kaggle Baseline Reference: [Janio Martinez Bachmann’s notebook](https://www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets/notebook)
