from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import shap

def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    return model

def evaluate_model(model, X, y):
    preds = model.predict(X)
    print(classification_report(y, preds))

# SHAP explainer
"""
def explain_model(model, X):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    shap.plots.beeswarm(shap_values)
"""

def explain_model(model, X):
    explainer = shap.TreeExplainer(model)  # TreeExplainer for RandomForest
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values[1], X)  # Index 1 for positive class
