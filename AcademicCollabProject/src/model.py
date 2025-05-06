from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import shap
import matplotlib.pyplot as plt
import time
import os
import joblib

def save_model(model, filename="randomforest_model.pkl"):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def load_model(filename="randomforest_model.pkl"):
    if os.path.exists(filename):
        start_time = time.time()
        
        model = joblib.load(filename)
        print(f"Loaded model from {filename}")

        elapsed = time.time() - start_time
        minutes, seconds = divmod(elapsed, 60)
        print(f"Time: {int(minutes):02d}:{int(seconds):02d}")
        return model
    else:
        print(f"Model file not found: {filename}")
        return None

def train_model(X, y):
    start_time = time.time()
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

    elapsed = time.time() - start_time
    minutes, seconds = divmod(elapsed, 60)
    print(f"Training completed in {int(minutes):02d}:{int(seconds):02d}")
    return model

def evaluate_model(model, X, y):
    start_time = time.time()
    
    preds = model.predict(X)
    print(classification_report(y, preds))
    elapsed = time.time() - start_time
    minutes, seconds = divmod(elapsed, 60)
    print(f"Evaluation time: {int(minutes):02d}:{int(seconds):02d}")

# SHAP explainer - Not applicable for Kmeans
"""
def explain_model(model, X):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    shap.plots.beeswarm(shap_values)
"""

def explain_model(model, X):
    start_time = time.time()
    explainer = shap.TreeExplainer(model)  # TreeExplainer for RandomForest
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values[1], X)  # Index 1 for positive class
    plt.savefig("shap_summary.png") # Save plot
    plt.close()
    # Time result
    elapsed = time.time() - start_time
    minutes, seconds = divmod(elapsed, 60)
    print(f"Explain time: {int(minutes):02d}:{int(seconds):02d}")