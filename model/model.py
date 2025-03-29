import os
import joblib
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from utlils.dataloader import load_dicom_images
from utlils.featureextracter import extract_features
from utlils.preprocess import preprocess_image
import pydicom
import numpy as np

# Paths and constants
main_dir = "data/Train/"
class_names = [f for f in sorted(os.listdir(main_dir)) if os.path.isdir(os.path.join(main_dir, f))]
model_path = "model/xgboost_model.pkl"

def train_model():
    # Load dataset
    X, y = load_dicom_images(main_dir, class_names)

    # Extract features
    X_features = extract_features(X)

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)

    # Train model
    clf = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
    clf.fit(X_scaled, y)

    # Save model and scaler
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump((clf, scaler, class_names), model_path)
    print("Model saved to", model_path)

def predict_dicom(image_path):
    # Load model and scaler
    clf, scaler, class_names = joblib.load(model_path)
    
    # Load and preprocess image
    dicom_data = pydicom.dcmread(image_path)
    img = dicom_data.pixel_array
    img = preprocess_image(img)
    
    # Extract features
    features = extract_features([img])
    
    # Normalize features
    features_scaled = scaler.transform(features)
    
    # Predict class
    class_index = clf.predict(features_scaled)[0]
    return class_names[class_index]
