# Fraud Detection in Financial Transactions

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model & Methodology](#model--methodology)
- [Architecture](#architecture)
- [Performance](#performance)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Web Application Interface](#web-application-interface)
- [Future Improvements](#future-improvements)

## Overview

This project implements an end-to-end fraud detection system for financial transactions. Using a **Random Forest Classifier** algorithm, the system classifies transactions as legitimate or fraudulent with ~87% accuracy. The solution includes a trained ML model, a Flask REST API backend, and an interactive HTML web interface for real-time predictions.

## Project Structure

```
fraud-detection/
├── model_training.ipynb        # Jupyter notebook with complete model development
├── api.py                       # Flask API server with prediction endpoints
├── data.csv                     # Training dataset (2,512 transactions)
├── svm_model.pkl                # Trained Random Forest model (binary)
├── scaler.pkl                   # StandardScaler for feature normalization
├── requirements.txt             # Python dependencies
├── templates/
│   └── index.html              # Web UI for predictions
└── README.md                    # This file
```

## Dataset

The model is trained on a financial transaction dataset (`data.csv`) with the following features:

| Feature | Type | Description |
|---------|------|-------------|
| `TransactionAmount` | Numerical | Amount of the transaction |
| `CustomerAge` | Numerical | Age of the customer |
| `AccountBalance` | Numerical | Account balance before transaction |
| `Channel` | Categorical | Transaction channel (Online, ATM, Branch) |
| `LoginAttempts` | Numerical | Number of login attempts before transaction |
| `is_fraud` | Target | Binary label (0 = Legitimate, 1 = Fraudulent) |

## Model & Methodology

### Data Understanding & Loading

The `Data` class loads the CSV dataset into a Pandas DataFrame and performs initial exploration including null value checks and descriptive statistics.

### Data Preprocessing

The `DataPreprocessing` class:
- **Feature Selection:** Selects relevant features for model training
- **Label Encoding:** Converts categorical `Channel` values to numerical: `{'ATM': 0, 'Online': 1, 'Branch': 2}`

### Exploratory Data Analysis (EDA)

Multiple EDA classes generate visualizations:
- **Univariate Analysis:** Distribution plots of individual features (`TransactionAmount`, `CustomerAge`)
- **Bivariate Analysis:** Relationship analysis between features and the fraud target variable

### Model Development

The `Model` class executes the ML pipeline:
- **Data Splitting:** 80% training / 20% testing with stratified sampling
- **Feature Scaling:** `StandardScaler` standardizes all features (mean=0, std=1)
- **Algorithm:** Random Forest Classifier with optimized hyperparameters
  - `n_estimators=300` - 300 decision trees
  - `max_depth=10` - Tree depth limit
  - `min_samples_leaf=2` - Minimum samples per leaf
  - `class_weight='balanced_subsample'` - Handles class imbalance
- **Threshold Tuning:** Probability threshold set to 0.62 for fraud predictions
- **Model Persistence:** `PickleData` class serializes trained model and scaler

### Model Evaluation

Performance metrics calculated on test set:
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)
- Accuracy Score

## Architecture

The project consists of two main components:

**1. Model Training Pipeline (`model_training.ipynb`)**
   - `Data`, `DataPreprocessing`, `Graph`, `UnivariateAnalysis`, `BivariateAnalysis` classes
   - `Model` class manages the Random Forest training pipeline
   - `PickleData` class serializes model and scaler

**2. Flask API & Web Interface (`api.py` + `index.html`)**
   - RESTful API endpoints for health checks and fraud prediction
   - CORS-enabled for cross-origin requests
   - HTML form interface for user input
   - Real-time prediction with risk classification

## Performance

**Model Performance Summary:**

**Test Set Accuracy:** 86.68% (~87%)

**Classification Report:**
```
              precision    recall  f1-score   support
           0       0.88      0.95      0.91       376
           1       0.81      0.61      0.70       127
    accuracy                           0.87       503
   macro avg       0.85      0.78      0.81       503
weighted avg       0.86      0.87      0.86       503
```

**Confusion Matrix:**
```
             Predicted Legitimate  Predicted Fraud
Actual Legitimate      358                18
Actual Fraud           49                 78
```

**Performance by Class:**

| Class | Metric | Value | Interpretation |
|-------|--------|-------|-----------------|
| **Legitimate (0)** | Precision | 88% | 88% of predicted legitimate transactions are actually legitimate |
| **Legitimate (0)** | Recall | 95% | Model catches 95% of all legitimate transactions |
| **Legitimate (0)** | F1-Score | 0.91 | Excellent overall performance on legitimate transactions |
| **Fraudulent (1)** | Precision | 81% | 81% of predicted fraud cases are actually fraudulent (low false positives) |
| **Fraudulent (1)** | Recall | 61% | Model identifies 61% of actual fraudulent transactions |
| **Fraudulent (1)** | F1-Score | 0.70 | Moderate overall performance on fraud detection |

**Model Strengths:**
- ✅ Excellent legitimate detection: 95% recall minimizes false negatives
- ✅ High fraud precision: 81% precision reduces false positives
- ✅ Balanced overall accuracy: 87%
- ✅ Handles class imbalance with balanced class weights

**Considerations:**
- ⚠️ Fraud recall at 61%: ~39% of fraudulent transactions may be missed
- ⚠️ Trade-off optimized for low false positives; may allow some fraud

## Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

**Required Packages:**
- `pandas` - Data manipulation and analysis
- `scikit-learn` - Machine learning algorithms and preprocessing
- `matplotlib` - Data visualization
- `flask` - Web API framework
- `flask-cors` - Cross-Origin Resource Sharing support

### Step 2: Prepare Model Files
Ensure the following files exist in the project directory:
- `svm_model.pkl` - Trained Random Forest model
- `scaler.pkl` - Feature scaler for normalization

If these files don't exist, run the model training notebook first:
```bash
jupyter notebook model_training.ipynb
```

## Usage

### Option 1: Run Flask API + Web Interface (Recommended)

**Start the Flask server:**
```bash
python api.py
```

**Access the application:**
- Open your browser and navigate to: `http://localhost:5000`
- Use the web form to enter transaction details
- Get real-time fraud risk prediction

### Option 2: Use REST API Directly

**Health Check:**
```bash
curl http://localhost:5000/health
```

**Make Prediction:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionAmount": 1500,
    "CustomerAge": 35,
    "AccountBalance": 50000,
    "Channel": "Online",
    "LoginAttempts": 2
  }'
```

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Web interface |
| `/api` | GET | API information and endpoints |
| `/health` | GET | Health check |
| `/predict` | POST | Fraud prediction |

### Predict Endpoint Details

**Request Format:**
```json
{
  "TransactionAmount": <number>,
  "CustomerAge": <number>,
  "AccountBalance": <number>,
  "Channel": "<Online|Branch|ATM>",
  "LoginAttempts": <number>
}
```

**Response Format (Fraud Detected):**
```json
{
  "prediction": 1,
  "is_fraud": true,
  "message": "High Risk: Fraud Detected!"
}
```

**Response Format (Legitimate):**
```json
{
  "prediction": 0,
  "is_fraud": false,
  "message": "Low Risk: Transaction is Legitimate"
}
```

## Web Application Interface

The interactive web interface (`index.html`) provides:
- **Input Form:** Enter transaction details (amount, age, balance, channel, login attempts)
- **Real-time Predictions:** Instant classification with visual feedback
- **Risk Indicators:** Color-coded results (Red for high risk 🚨, Green for low risk ✅)
- **Responsive Design:** Works on desktop and mobile browsers

## Future Improvements

### Model Enhancement
- **Improve Fraud Recall:** Implement SMOTE (Synthetic Minority Over-sampling) to generate synthetic fraud examples
- **Advanced Algorithms:** Test XGBoost, LightGBM, and Neural Networks for potentially higher performance
- **Feature Engineering:** 
  - Transaction frequency analysis
  - Time-of-day patterns
  - Historical transaction behavior
  - Device fingerprinting

### Production Deployment
- **Cloud Deployment:** Deploy to AWS, GCP, or Azure for public access
- **Database Integration:** Store transaction history and predictions
- **Authentication:** Add user accounts and API key authentication
- **Model Versioning:** Implement version control for model updates

### Monitoring & Maintenance
- **Performance Tracking:** Monitor model accuracy in production
- **Anomaly Alerts:** Alert system for sudden performance degradation
- **Retraining Pipeline:** Automated model retraining on new data
- **A/B Testing:** Compare model versions for continuous improvement

### Advanced Features
- **Explainability:** Add SHAP values to explain individual predictions
- **Batch Processing:** Support bulk fraud detection for large transaction volumes
- **Real-time Updates:** Stream processing for instant fraud alerts
- **Custom Thresholds:** Allow users to adjust fraud probability threshold