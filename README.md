# 🧠 Multi-Disease Prediction System

A simple and interactive web application for predicting the likelihood of **Diabetes**, **Heart Disease**, and **Parkinson’s Disease** using machine learning.

This project combines a user-friendly frontend with a robust Python Flask backend powered by pre-trained SVM models. It’s designed for educational and demonstration purposes to showcase how AI can assist in basic health screening.

---

## 🚀 Features

- 🎯 **Multi-Disease Support**  
  Predicts Diabetes, Heart Disease, or Parkinson’s Disease from user-input health parameters.

- 🧩 **Dynamic Input Forms**  
  Automatically adapts the form fields based on the selected disease.

- 🧠 **ML-Driven Predictions**  
  Utilizes pre-trained **Support Vector Machine (SVM)** models for accurate predictions.

- ⚙️ **Built-in Preprocessing**  
  Handles missing values, outlier detection, and feature scaling on the backend for consistent and reliable input handling.

- 💾 **Model Persistence**  
  Machine learning models and preprocessing objects are stored using **joblib**, ensuring fast loading and efficient predictions.

---

## 🛠️ Technologies Used

### 🔹 Frontend
- **HTML5**  
- **Tailwind CSS** – for modern UI styling  
- **JavaScript** – to create dynamic user interactions

### 🔹 Backend
- **Python 3.x**  
- **Flask** – lightweight web framework  
- **Flask-CORS** – enables cross-origin requests  
- **scikit-learn** – for ML models and preprocessing  
- **imbalanced-learn** – for SMOTE oversampling  
- **pandas** – for data manipulation  
- **joblib** – for model and preprocessing object serialization

---

## ⚠️ Disclaimer

> **This system is for educational and demonstration purposes only.**  
> The predictions are based on simplified models trained on publicly available datasets and should **not** be used for actual medical diagnosis or treatment.  
> Always consult a certified healthcare provider for any medical concerns or decisions.



