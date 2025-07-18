# app.py

import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS # To handle Cross-Origin Resource Sharing

# --- Set environment variables to mitigate threading issues (important for model loading) ---
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

app = Flask(__name__)
CORS(app) # Enable CORS for all routes, allowing frontend to access

# --- Define paths to your saved models and preprocessing objects ---
# Make sure these paths are correct relative to where app.py is run
MODELS_DIR = '.' # Assuming models are in the same directory as app.py

# Diabetes Model Assets
DIABETES_MODEL_PATH = os.path.join(MODELS_DIR, 'diabetes_svm_model.joblib')
DIABETES_SCALER_PATH = os.path.join(MODELS_DIR, 'diabetes_scaler.joblib')
DIABETES_MEANS_PATH = os.path.join(MODELS_DIR, 'diabetes_train_means.joblib')
DIABETES_IQR_BOUNDS_PATH = os.path.join(MODELS_DIR, 'diabetes_train_iqr_bounds.joblib')

# Heart Disease Model Assets
HEART_MODEL_PATH = os.path.join(MODELS_DIR, 'heart_disease_svm_model.joblib')
HEART_SCALER_PATH = os.path.join(MODELS_DIR, 'heart_disease_scaler.joblib')
HEART_IQR_BOUNDS_PATH = os.path.join(MODELS_DIR, 'heart_disease_train_iqr_bounds.joblib')

# Parkinson's Disease Model Assets
PARKINSONS_MODEL_PATH = os.path.join(MODELS_DIR, 'parkinsons_svm_model.joblib')
PARKINSONS_SCALER_PATH = os.path.join(MODELS_DIR, 'parkinsons_scaler.joblib')
PARKINSONS_IQR_BOUNDS_PATH = os.path.join(MODELS_DIR, 'parkinsons_train_iqr_bounds.joblib')

# --- Load all models and preprocessing objects on application startup ---
# This ensures they are loaded once and available for all requests
try:
    # Diabetes
    diabetes_model = joblib.load(DIABETES_MODEL_PATH)
    diabetes_scaler = joblib.load(DIABETES_SCALER_PATH)
    diabetes_train_means = joblib.load(DIABETES_MEANS_PATH)
    diabetes_train_iqr_bounds = joblib.load(DIABETES_IQR_BOUNDS_PATH)
    print("Diabetes models and preprocessing objects loaded successfully.")

    # Heart Disease
    heart_model = joblib.load(HEART_MODEL_PATH)
    heart_scaler = joblib.load(HEART_SCALER_PATH)
    heart_train_iqr_bounds = joblib.load(HEART_IQR_BOUNDS_PATH)
    print("Heart Disease models and preprocessing objects loaded successfully.")

    # Parkinson's Disease
    parkinsons_model = joblib.load(PARKINSONS_MODEL_PATH)
    parkinsons_scaler = joblib.load(PARKINSONS_SCALER_PATH)
    parkinsons_train_iqr_bounds = joblib.load(PARKINSONS_IQR_BOUNDS_PATH)
    print("Parkinson's Disease models and preprocessing objects loaded successfully.")

except FileNotFoundError as e:
    print(f"Error loading model files: {e}. Make sure all .joblib files are in the same directory as app.py")
    exit() # Exit if models can't be loaded, as the app won't function

# --- Preprocessing Function (Replicated from your training scripts) ---
# This function must be identical to the one used during training for consistency
def preprocess_data(data_dict, scaler, train_iqr_bounds, cols_to_replace_zero=None, train_means=None):
    # Convert dictionary to DataFrame
    # Ensure column order matches training data
    if 'Pregnancies' in data_dict: # Diabetes data
        feature_order = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    elif 'age' in data_dict: # Heart data
        feature_order = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    elif 'MDVP:Fo(Hz)' in data_dict: # Parkinson's data
        feature_order = [
            'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
            'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:RAP', 'MDVP:Shimmer',
            'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ',
            'Shimmer:DD', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
        ]
    else:
        raise ValueError("Unknown data format. Cannot determine feature order.")

    df = pd.DataFrame([data_dict], columns=feature_order)

    # Handle missing values (0s replaced with training means) - specific to Diabetes
    if cols_to_replace_zero and train_means:
        for col in cols_to_replace_zero:
            df[col] = df[col].replace(0, train_means[col])

    # Outlier Treatment (Capping using training IQR bounds)
    for column in df.columns:
        if column in train_iqr_bounds: # Ensure bounds exist for the column
            lower_bound = train_iqr_bounds[column]['lower']
            upper_bound = train_iqr_bounds[column]['upper']
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

    # Feature Scaling (using the loaded scaler)
    scaled_data = scaler.transform(df)
    processed_df_scaled = pd.DataFrame(scaled_data, columns=df.columns)

    return processed_df_scaled

# --- API Endpoints ---

@app.route('/')
def home():
    return "Disease Prediction Backend is running. Access the frontend HTML file in your browser."

@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    try:
        data = request.get_json(force=True)
        print(f"Received diabetes data: {data}")

        # Define columns that might have 0s representing missing values for diabetes
        cols_to_replace_zero_diabetes = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

        processed_data = preprocess_data(
            data,
            scaler=diabetes_scaler,
            train_iqr_bounds=diabetes_train_iqr_bounds,
            cols_to_replace_zero=cols_to_replace_zero_diabetes,
            train_means=diabetes_train_means
        )

        prediction = diabetes_model.predict(processed_data)[0]
        probability = diabetes_model.predict_proba(processed_data)[0].tolist() # Convert to list for JSON

        return jsonify({'prediction': int(prediction), 'probability': probability})
    except Exception as e:
        print(f"Error in diabetes prediction: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/predict_heart_disease', methods=['POST'])
def predict_heart_disease():
    try:
        data = request.get_json(force=True)
        print(f"Received heart disease data: {data}")

        processed_data = preprocess_data(
            data,
            scaler=heart_scaler,
            train_iqr_bounds=heart_train_iqr_bounds
        )

        prediction = heart_model.predict(processed_data)[0]
        probability = heart_model.predict_proba(processed_data)[0].tolist()

        return jsonify({'prediction': int(prediction), 'probability': probability})
    except Exception as e:
        print(f"Error in heart disease prediction: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/predict_parkinsons', methods=['POST'])
def predict_parkinsons():
    try:
        data = request.get_json(force=True)
        print(f"Received parkinsons data: {data}")

        processed_data = preprocess_data(
            data,
            scaler=parkinsons_scaler,
            train_iqr_bounds=parkinsons_train_iqr_bounds
        )

        prediction = parkinsons_model.predict(processed_data)[0]
        probability = parkinsons_model.predict_proba(processed_data)[0].tolist()

        return jsonify({'prediction': int(prediction), 'probability': probability})
    except Exception as e:
        print(f"Error in parkinsons prediction: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Run the Flask app
    # debug=True allows for automatic reloading on code changes and provides more detailed error messages
    app.run(debug=True, port=5000)
