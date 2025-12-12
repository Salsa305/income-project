import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score, 
    fbeta_score, f1_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from typing import Dict

def run_svm(
    X_train, X_test, y_train, y_test,
    preprocessor,
    kernel='linear',
    C=1.0,
    class_weight=None,
    probability=True
) -> Dict:
    """
    Train and evaluate an SVM model using the default 0.5 threshold, and returns 
    all standard and goal-aligned metrics, including PR-AUC.
    """
    if not probability:
         raise ValueError("SVC must be run with probability=True to generate probability scores for PR-AUC/ROC-AUC.")
         
    model = SVC(
        kernel=kernel,
        C=C,
        class_weight=class_weight,
        probability=probability,
        random_state=42
    )

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    pipe.fit(X_train, y_train)
    
    # --- 1. Predictions at Default Threshold (0.5) ---
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    
    # --- 2. Calculate Metrics ---
    
    # Ranking Metrics (Threshold-Independent)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    
    # Threshold-Dependent Metrics (at 0.5)
    # Note: We use fbeta_score to calculate F0.5, ensuring the naming is consistent
    # with the logistic regression file, but using the predictions at 0.5.
    f05_score_at_05 = fbeta_score(y_test, y_pred, beta=0.5, zero_division=0)
    
    # --- 3. Generate Classification Report (Full Standard Metrics) ---
    # This includes Accuracy, F1-score, Support, etc., for both classes
    report_str = classification_report(y_test, y_pred, digits=4, output_dict=False)
    
    # --- 4. Append Goal-Aligned Metrics to the Report String ---
    
    # Create a small summary DataFrame for the ranking metrics
    summary_data = {
        'PR-AUC': [pr_auc],
        'ROC-AUC': [roc_auc]
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
        "f05_score": f05_score_at_05,
        "report": final_report
    }


def cross_validate_svm(pipeline, X, y, cv=5, scoring='average_precision'):
    """
    Perform stratified k-fold cross-validation for an SVM pipeline, 
    now defaulting to 'average_precision' (PR-AUC) for tuning.
    Returns an array of scores.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=skf, scoring=scoring)
    return scores