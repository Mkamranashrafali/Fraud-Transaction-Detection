# Fraud Detection in Financial Transactions

## Table of Contents
- [Fraud Detection in Financial Transactions](#fraud-detection-in-financial-transactions)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Project Structure](#project-structure)
  - [Dataset](#dataset)
  - [Model \& Methodology](#model--methodology)
    - [Data Understanding \& Loading](#data-understanding--loading)
    - [Data Preprocessing](#data-preprocessing)
    - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
    - [Model Development](#model-development)
    - [Model Evaluation](#model-evaluation)
  - [Architecture](#architecture)
  - [Data](#data)
  - [Methodology](#methodology)
    - [Data Understanding \& Loading](#data-understanding--loading-1)
    - [Data Preprocessing](#data-preprocessing-1)
    - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda-1)
    - [Model Development](#model-development-1)
    - [Model Evaluation](#model-evaluation-1)
  - [Web Application Interface](#web-application-interface)
  - [Results](#results)
  - [How to Run](#how-to-run)
  - [Conclusion \& Future Work](#conclusion--future-work)

## Overview

This project implements an end-to-end fraud detection system for financial transactions. Using a **Support Vector Machine (SVlsM)** algorithm, the system classifies transactions as legitimate or fraudulent with ~81% accuracy. The solution includes a trained ML model, a Flask REST API backend, and an interactive HTML web interface for real-time predictions.

## Project Structure

```
fraud-detection/
├── model_training.ipynb        # Jupyter notebook with complete model development
├── api.py                       # Flask API server with prediction endpoints
├── data.csv                     # Training dataset
├── svm_model.pkl                # Trained SVM model (binary)
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
- **Label Encoding:** Converts categorical `Channel` values to numerical: `{'Online': 0, 'Branch': 1, 'ATM': 2}`

### Exploratory Data Analysis (EDA)

Multiple EDA classes generate visualizations:
- **Univariate Analysis:** Distribution plots of individual features (`TransactionAmount`, `CustomerAge`)
- **Bivariate Analysis:** Relationship analysis between features and the fraud target variable

### Model Development

The `Model` class executes the ML pipeline:
- **Data Splitting:** 80% training / 20% testing with stratified sampling
- **Feature Scaling:** `StandardScaler` standardizes all features (mean=0, std=1)
- **Algorithm:** Support Vector Classifier (SVC) with linear kernel
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
   - `Model` class manages the SVM training pipeline
   - `PickleData` class serializes model and scaler

**2. Flask API & Web Interface (`api.py` + `index.html`)**
   - RESTful API endpoints for health checks and fraud prediction
   - CORS-enabled for cross-origin requests
   - HTML form interface for user input
   - Real-time prediction with risk classification

## Data

The model is trained on a financial transaction dataset (`data.csv`) using the following key features:

*   **Numerical Features:** `TransactionAmount`, `CustomerAge`, `AccountBalance`, `LoginAttempts`.
*   **Categorical Feature:** `Channel` (encoded into `ChannelEncoded`).
*   **Target Variable:** `is_fraud` (a binary variable where `1` indicates a fraudulent transaction and `0` indicates a legitimate one).

## Methodology

### Data Understanding & Loading

The process begins by loading the `data.csv` dataset into a Pandas DataFrame. Initial exploration is performed to understand the data's structure, check for null values, and generate descriptive statistics.

### Data Preprocessing

The `DataPreprocessing` class executes the following key steps:
*   **Feature Selection:** A subset of relevant features is chosen for model training.
*   **Label Encoding:** The categorical `Channel` column is converted into numerical representations (`{'ATM': 0, 'Online': 1, 'Branch': 2}`), making it suitable for the SVM algorithm.

### Exploratory Data Analysis (EDA)

Visualizations are created to gain insights from the data:
*   **Univariate Analysis:** Distributions of individual features like `TransactionAmount` and `CustomerAge` are plotted to understand their spread.
*   **Bivariate Analysis:** The relationship between features and the target variable (`is_fraud`) is analyzed to identify patterns associated with fraudulent activities.

### Model Development

The core machine learning pipeline is executed by the `Model` class:
*   **Data Splitting:** The dataset is split into training (80%) and testing (20%) sets. Stratified sampling is used to maintain the class distribution in both sets.
*   **Feature Scaling:** `StandardScaler` is applied to the training and testing data. This standardizes features to have a mean of 0 and a standard deviation of 1, which is crucial for SVM performance.
*   **Model Training:** A **Support Vector Classifier (SVC)** with a linear kernel is trained on the scaled training data.

### Model Evaluation

The trained model's performance is assessed on the unseen test data using standard metrics:
*   **Confusion Matrix:** To see the counts of correct and incorrect predictions.
*   **Classification Report:** To review precision, recall, and F1-score for each class.
*   **Accuracy Score:** To get the overall percentage of correct classifications.

## Performance

**Model Performance Summary:**

```
              precision    recall  f1-score   support
           0       0.83      0.94      0.88       376
           1       0.70      0.43      0.53       127
    accuracy                           0.81       503
```

**Key Metrics:**
*   **Overall Accuracy:** ~**81%** - The model correctly classifies 81% of transactions
*   **Fraud Detection (Class 1):**
    *   **Precision:** 70% - When the model predicts fraud, it's correct 70% of the time
    *   **Recall:** 43% - The model identifies 43% of actual fraudulent transactions
*   **Legitimate Detection (Class 0):**
    *   **Precision:** 83% - High accuracy for legitimate transactions
    *   **Recall:** 94% - Catches most legitimate transactions

## Installation & Setup

### Prerequisites
- Python 3.7+
- pip package manager

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

**Dependencies:**
- `pandas` - Data manipulation
- `scikit-learn` - ML algorithms and preprocessing
- `matplotlib` - Data visualization
- `flask` - Web API framework
- `flask-cors` - Cross-Origin Resource Sharing support
- `streamlit` (optional) - For alternative UI

### Step 2: Prepare Model Files
Ensure the following files exist in the project directory:
- `svm_model.pkl` - Trained SVM model
- `scaler.pkl` - Feature scaler for normalization

If these files don't exist, run the model training notebook first to generate them.

## Usage

### Option 1: Run Flask API + Web Interface (Recommended)

1. Start the Flask server:
```bash
python api.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Use the web form to input transaction details and get real-time predictions

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

Planned enhancements to boost model performance and functionality:

*   **Improve Model Recall:** 
    - Implement SMOTE (Synthetic Minority Over-sampling Technique)
    - Use class weights to balance the imbalanced dataset
    - Improve detection of fraudulent cases

*   **Advanced Models:**
    - Experiment with XGBoost or LightGBM for higher performance
    - Test ensemble methods combining multiple algorithms
    - Implement neural networks with deep learning

*   **Feature Engineering:**
    - Develop transaction frequency features
    - Add time-of-day analysis
    - Incorporate historical transaction patterns

*   **Production Deployment:**
    - Deploy to cloud platforms (AWS, GCP, Azure)
    - Add database integration for transaction logging
    - Implement model versioning and A/B testing
    - Add authentication and authorization

*   **Monitoring:**
    - Track model performance in production
    - Alert system for anomalies
    - Model retraining pipeline