import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    fbeta_score, precision_recall_curve, precision_score, recall_score,
    accuracy_score, classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from typing import Dict

def run_logistic(
    X_train, X_test, y_train, y_test,
    preprocessor,
    penalty="l2",
    C=1.0,
    class_weight=None
) -> Dict:
    """
    Runs Logistic Regression using the default 0.5 threshold, and returns 
    all standard and goal-aligned metrics, including PR-AUC.
    """
    model = LogisticRegression(
        penalty=penalty,
        C=C,
        solver="liblinear",
        max_iter=1000,
        class_weight=class_weight
    )

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    
    # --- 1. Predictions at Default Threshold (0.5) ---
    # The default threshold is 0.5, implemented by model.predict()
    y_pred = pipe.predict(X_test)
    
    # --- 2. Calculate Metrics ---
    
    # Ranking Metrics (Threshold-Independent)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    
    # Threshold-Dependent Metrics (at 0.5)
    f05_score_at_05 = fbeta_score(y_test, y_pred, beta=0.5, zero_division=0)
    
    # --- 3. Generate Classification Report (Full Standard Metrics) ---
    # This includes Accuracy, F1-score, Support, etc., for both classes
    report_str = classification_report(y_test, y_pred, digits=4, output_dict=False)
    
    # --- 4. Append Goal-Aligned Metrics to the Report String ---
    
    # Create a small summary DataFrame for the ranking metrics
    summary_data = {
        'PR-AUC': [pr_auc],
        'ROC-AUC': [roc_auc],
        'f05_score': [f05_score_at_05]
    }
    df_summary = pd.DataFrame(summary_data, index=['Ranking Metrics']).T 
    summary_str = df_summary.to_string(float_format='%.4f')
    
    # Combine the standard report with the ranking metrics
    final_report = f"""
Standard Classification Report (@ Threshold 0.5):
{report_str}

---------------------------------------------
Goal-Aligned Ranking Metrics (Threshold-Independent):

{summary_str}
---------------------------------------------
"""
    
    # --- 5. Return in the EXACT requested structure ---
    return {
        "pipeline": pipe,
        "auc": roc_auc,
        "pr_auc": pr_auc,
        "f05_score": f05_score_at_05, # F0.5 at the default 0.5 threshold
        "report": final_report
    }

def cross_validate_logistic(pipeline, X, y, cv=5, scoring='average_precision'):
    """
    Recommends 'average_precision' (PR-AUC) as the default scoring metric
    for cross-validation in this precision-first problem.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=skf, scoring=scoring)
    return scores