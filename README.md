# 🧠 Multi-Disease Prediction System

A web-based machine learning application for predicting the likelihood of **Diabetes**, **Heart Disease**, and **Parkinson's Disease** using health parameters provided by the user.

This system combines an intuitive frontend with a Python Flask backend to deliver predictions from pre-trained Support Vector Machine (SVM) models. It demonstrates the application of machine learning to real-world health prediction scenarios.

---

## 📌 Overview

The Multi-Disease Prediction System allows users to choose from one of three diseases and input relevant medical parameters. The backend performs necessary preprocessing steps (imputation, outlier handling, scaling) and returns a prediction based on trained machine learning models.

---

## 🚀 Features

- **🔄 Dynamic Disease Selection:**  
  A single-page web interface where users can choose the disease (Diabetes, Heart Disease, or Parkinson's) from a dropdown menu.

- **🧠 Machine Learning Predictions:**  
  Uses **SVM (Support Vector Machine)** classifiers trained on medical datasets for each disease.

- **🛠 Backend Preprocessing:**  
  Each model pipeline includes:
  - Missing value imputation using training means  
  - Outlier treatment using IQR bounds  
  - Feature scaling with `StandardScaler`

- **📦 Model Persistence with `joblib`:**  
  All trained models and preprocessing objects are saved as `.joblib` files and loaded at runtime for fast, reliable predictions.

- **🌐 REST API with Flask:**  
  A Python Flask backend exposes endpoints that receive JSON-formatted input and return model predictions.

---

## 🧠 Disease Models

Each disease prediction module has its own trained SVM model and associated preprocessing artifacts:

### ✅ Diabetes
- Dataset: `diabetes.csv`
- Model: `diabetes_svm_model.joblib`
- Scaler: `diabetes_scaler.joblib`
- Imputer Mean: `diabetes_train_means.joblib`
- IQR Bounds: `diabetes_train_iqr_bounds.joblib`

### ❤️ Heart Disease
- Dataset: `heart.csv`
- Model: `heart_disease_svm_model.joblib`
- Scaler: `heart_disease_scaler.joblib`
- IQR Bounds: `heart_disease_train_iqr_bounds.joblib`

### 🧠 Parkinson's Disease
- Dataset: `parkinsons.csv`
- Model: `parkinsons_svm_model.joblib`
- Scaler: `parkinsons_scaler.joblib`
- IQR Bounds: `parkinsons_train_iqr_bounds.joblib`

---

## 🌐 Technologies Used

### Frontend
- **HTML5**
- **Tailwind CSS** – for responsive design and modern UI
- **JavaScript** – for dynamic forms and API interaction

### Backend
- **Python 3.x**
- **Flask** – micro web framework
- **Flask-CORS** – enables cross-origin requests from the frontend
- **scikit-learn** – for SVM models and preprocessing
- **imbalanced-learn** – used for SMOTE oversampling (during model training)
- **pandas** – for data manipulation
- **joblib** – for serializing ML models and preprocessing pipelines

---

## ⚠️ Disclaimer

> This system is built **strictly for educational and demonstration purposes**.  
> The predictions are generated from simplified machine learning models trained on publicly available datasets.  
> **It is not intended for clinical use or real-world medical decision-making.**  
> Always consult a licensed healthcare professional for any medical concerns.

---


