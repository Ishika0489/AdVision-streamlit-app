import os
import joblib
import pandas as pd

def load_model(path='models/mlr_model.joblib'):
    """
    Load a trained model from a .joblib file.

    Parameters:
        path (str): Path to the saved model file.

    Returns:
        model: The loaded machine learning model.
    """
    print("🔍 Attempting to load model from:", path)
    print("📂 Current working directory:", os.getcwd())
    print("📁 Contents of models/:", os.listdir("models") if os.path.exists("models") else "models/ folder not found!")

    if not os.path.exists(path):
        raise Exception(f"❌ Model file not found at: {path}")

    model = joblib.load(path)
    print("✅ Model loaded successfully!")
    return model
