# FININT — Financial Intelligence & Predictive Analytics System

**An end-to-end machine learning project for financial transaction fraud detection, explainable AI, and investigation intelligence.**

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange)
![SHAP](https://img.shields.io/badge/Explainable%20AI-SHAP-purple)
![Project](https://img.shields.io/badge/Project-Portfolio%20Research-success)

## 1. Project Overview

FININT (Financial Intelligence & Predictive Analytics System) is a data science and machine learning project designed to analyse financial transactions, identify potentially fraudulent activities, and generate risk-based investigation alerts.

The project explores how predictive modelling, transaction behaviour analysis, account relationship analysis, and explainable artificial intelligence can support financial crime detection and investigation workflows.

Using the PaySim synthetic mobile-money transaction dataset, FININT implements an analytical pipeline covering data exploration, feature engineering, machine learning, model evaluation, explainability, and investigation prioritisation.

The project also generates an operational-style investigation register to demonstrate how machine learning predictions can be translated into structured investigation workflows.

## 2. Project Objectives

The main objectives are to:

- Analyse large-scale financial transaction data.
- Identify patterns associated with fraudulent transactions.
- Examine sender and recipient transaction behaviour.
- Engineer transaction history and velocity features.
- Develop and compare machine learning classification models.
- Evaluate model performance under severe class imbalance.
- Apply explainable AI to understand model predictions.
- Generate risk-based transaction alerts.
- Prioritise flagged transactions for investigation.
- Produce analytical reports and visualisations.

## 3. Dataset

FININT uses the PaySim synthetic mobile-money transaction dataset.

The original dataset contains approximately 6.36 million transactions and 11 columns.

**Dataset characteristics:**

| Attribute | Description |
|---|---|
| Records | 6,362,620 |
| Original columns | 11 |
| Transaction types | CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER |
| Target variable | isFraud |
| Legitimate transactions | 6,354,407 |
| Fraudulent transactions | 8,213 |
| Fraud prevalence | Approximately 0.13% |

The dataset includes transaction amounts, transaction types, sender and recipient identifiers, account balances, timestamps represented by simulation steps, and fraud labels.

The dataset is synthetic and is intended for research and experimentation. Results should not be interpreted as evidence of production performance on real financial institution data.

## 4. Technologies and Tools

| Technology | Application |
|---|---|
| Python | Main programming language |
| Jupyter Notebook | Interactive data science development |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computation |
| Matplotlib | Data visualisation |
| Seaborn | Statistical visualisation |
| Scikit-learn | Machine learning and evaluation |
| SHAP | Model explainability |
| NetworkX | Transaction relationship and network analysis |

## 5. Project Workflow

### Phase 1 — Data Exploration and Preparation

- Dataset inspection and structural analysis.
- Missing-value and duplicate checks.
- Fraud distribution analysis.
- Transaction amount and transaction type analysis.
- Sender and recipient balance examination.
- Fraud pattern exploration.

### Phase 2 — Transaction Behaviour and Feature Engineering

- Sender account aggregation.
- Destination account analysis.
- Repeated recipient investigation.
- Fraudulent sender-recipient relationship analysis.
- Directed transaction network construction.
- Historical recipient transaction analysis.
- Transaction velocity features across different time windows.
- Historical transaction amount and frequency features.

### Phase 3 — Machine Learning

Two classification approaches were developed and evaluated.

**1. Logistic Regression (SGD Classifier)**

Used as a scalable baseline model for large-scale transaction classification.

**2. Random Forest Classifier**

Developed as the primary fraud detection model using majority-class downsampling and class weighting.

The Random Forest was trained using a reduced training sample containing all available fraudulent training observations and a sampled subset of legitimate transactions.

### Phase 4 — Model Evaluation

A chronological training, validation, and testing strategy was used to preserve transaction time ordering.

| Dataset | Records |
|---|---:|
| Training | 4,463,587 |
| Validation | 943,289 |
| Testing | 955,744 |

The Random Forest decision threshold was selected using validation F1-score.

The final threshold was 0.95.

Model evaluation included:

- Confusion matrix.
- Precision.
- Recall.
- F1-score.
- Accuracy.
- ROC-AUC.
- Precision-Recall AUC.
- Threshold comparison.

### Phase 5 — Explainable AI

SHAP (SHapley Additive exPlanations) was incorporated to investigate the features contributing to model predictions.

The analysis examined:

- Global feature importance.
- Individual transaction explanations.
- Correctly detected fraudulent transactions.
- Missed fraudulent transactions.
- Positive and negative feature contributions.

The analysis highlighted sender balance, transaction amount, transaction type, and destination balance among the influential model features.

### Phase 6 — Investigation Intelligence

FININT converts model predictions into an investigation-oriented workflow.

The system generates a register containing:

- Investigation ID.
- Transaction ID.
- Transaction type.
- Transaction amount.
- Sender and recipient balance information.
- Fraud model score.
- Investigation priority.
- Investigation status.
- Assigned investigator.
- Review date.
- Investigation notes.
- Case outcome.
- Recommended action.

Four investigation priority categories were established:

| Priority | Model score range |
|---|---|
| Priority 1 | 0.99–1.00 |
| Priority 2 | 0.98–0.99 |
| Priority 3 | 0.97–0.98 |
| Priority 4 | 0.95–0.97 |

These are provisional model-score bands for organising the investigation workload, not independently validated financial crime severity classifications.

## 6. Model Performance

The Random Forest model was evaluated on 955,744 previously unseen chronological test transactions.

### Final test results

| Evaluation Metric | Result |
|---|---:|
| Precision | 93.65% |
| Recall | 64.34% |
| F1-score | 76.27% |
| Accuracy | 99.83% |
| ROC-AUC | 99.66% |
| PR-AUC | 81.89% |
| Decision threshold | 0.95 |

### Confusion Matrix

| Actual / Predicted | Legitimate | Fraudulent |
|---|---:|---:|
| Legitimate | 951,559 | 175 |
| Fraudulent | 1,430 | 2,580 |

### Investigation Results

| Metric | Result |
|---|---:|
| Transactions evaluated | 955,744 |
| Actual fraudulent transactions | 4,010 |
| Investigation alerts generated | 2,755 |
| Correctly detected fraud | 2,580 |
| False positive alerts | 175 |
| Missed fraudulent transactions | 1,430 |

The model achieved high alert precision, meaning that most flagged transactions were genuinely fraudulent in the test dataset. However, it did not identify every fraudulent transaction.

### Transaction-Type Performance

| Metric | CASH_OUT | TRANSFER |
|---|---:|---:|
| Precision | 100.00% | 91.85% |
| Recall | 30.37% | 98.30% |
| F1-score | 46.60% | 94.97% |
| Detected fraud | 609 | 1,971 |
| Missed fraud | 1,396 | 34 |

An important finding was the difference in detection performance between CASH_OUT and TRANSFER transactions.

This highlights the importance of evaluating fraud detection models across transaction categories rather than relying exclusively on overall accuracy.

**Note:** The model was trained on downsampled and class-weighted data. Its output scores are not calibrated real-world fraud probabilities. The 0.95 threshold is a selected model-score cutoff, not a claim that a transaction has a 95% actual probability of fraud.

## 7. Dashboard and Analytical Outputs

FININT generates visual reports to support the interpretation of its analytical findings.

The dashboard analysis covers:

- Fraud detection outcomes.
- Investigation priority distribution.
- Investigation workload by transaction type.
- Investigation priority by transaction type.
- Fraud composition of flagged transactions.
- Precision by transaction type.
- Fraud detection performance by transaction type.
- Detected versus missed fraudulent transactions.

Charts are exported in high-resolution PNG format, while supporting analytical tables are exported as CSV files.

## 8. Project Structure

```text
FININT/
│
├── datasets/
│   └── paysim.csv
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── outputs/
│   ├── FININT_Investigation_Register.csv
│   ├── FININT_Model_Evaluation_Results.csv
│   ├── FININT_Workload_By_Transaction_Type.png
│   ├── FININT_Priority_By_Transaction_Type.png
│   ├── FININT_Transaction_Type_Workload.csv
│   ├── FININT_Transaction_Type_Priority.csv
│   ├── FININT_Fraud_Composition_By_Type.png
│   ├── FININT_Fraud_Composition_By_Type.csv
│   ├── FININT_Precision_By_Transaction_Type.csv
│   ├── FININT_Performance_By_Transaction_Type.png
│   ├── FININT_Detected_vs_Missed_Fraud.png
│   └── FININT_Performance_By_Transaction_Type.csv
│
├── README.md
└── requirements.txt
```

The directory listing represents the current analytical deliverables and the intended project structure. Additional model and dashboard modules can be added as development progresses.

## 9. Installation and Setup

### Prerequisites

- Python 3.x
- Jupyter Notebook
- pip

### Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

```bash
cd FININT
```

### Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the notebook in the `notebooks` directory and execute the analysis cells in sequence.

The PaySim dataset should be downloaded separately and placed in the `datasets` directory. Update the notebook's dataset path if necessary.

## 10. Requirements

The main Python dependencies include:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
shap
networkx
jupyter
ipykernel
```

## 11. Limitations and Future Improvements

Although FININT demonstrates a complete analytical fraud detection workflow, several areas require further development.

**Current limitations:**

- PaySim is synthetic and may not fully represent real financial transaction behaviour.
- Temporal changes in fraud prevalence indicate distribution shift.
- The model misses a substantial proportion of CASH_OUT fraud.
- Model scores have not undergone probability calibration.
- Investigation priorities are provisional and require operational validation.
- Account relationship patterns in the synthetic dataset do not establish criminal intent or coordinated activity.

**Planned improvements:**

- Investigate and improve CASH_OUT fraud detection.
- Explore probability calibration.
- Evaluate additional classification algorithms.
- Implement more advanced anomaly detection techniques.
- Expand explainability and model monitoring.
- Develop an interactive financial intelligence dashboard.
- Introduce investigation case management and reporting.
- Explore additional datasets and real-world validation opportunities.

## 12. Disclaimer

FININT is an educational and portfolio research project developed using synthetic financial transaction data.

It is not a deployed banking fraud detection system and should not be used to make real financial, regulatory, investigative, or law enforcement decisions without appropriate validation, security, governance, and human review.

## 13. Author

**Abdullahi Mansur Musawa**

Data Science | Data Analytics | Machine Learning | Financial Intelligence | Cybersecurity

Interested in building data-driven solutions for financial intelligence, fraud analytics, predictive modelling, and business decision-making.

**Project:** FININT — Financial Intelligence & Predictive Analytics System
