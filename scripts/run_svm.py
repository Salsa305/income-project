# scripts/run_svm.py

import numpy as np
from src.data import load_adult, preprocess_adult
from src.models.svm import run_svm, cross_validate_svm
from src.preprocessing import make_preprocessor

def main(sample_fraction=1.0, run_cv=True):
    print("="*80)
    print("LOADING DATA")
    print("="*80)
    
    X, y = load_adult()
    
    # Optional: reduce dataset for faster experimentation
    if sample_fraction < 1.0:
        X = X.sample(frac=sample_fraction, random_state=42)
        y = y.loc[X.index]
        print(f"Using {len(X)} samples for quick experimentation")
    
    X_train, X_test, y_train, y_test, cat_cols, num_cols = preprocess_adult(X, y)
    
    # Preprocessor from preprocessing.py
    preprocessor = make_preprocessor(cat_cols, num_cols)
    
    # Baseline Linear SVM
    print("="*80)
    print("LINEAR SVM BASELINE")
    linear_results = run_svm(X_train, X_test, y_train, y_test, preprocessor, kernel='linear')
    print("ROC-AUC:", linear_results["auc"])
    print(linear_results["report"])
    
    # Baseline RBF SVM
    print("="*80)
    print("RBF SVM BASELINE")
    rbf_results = run_svm(X_train, X_test, y_train, y_test, preprocessor, kernel='rbf')
    print("ROC-AUC:", rbf_results["auc"])
    print(rbf_results["report"])
    
    # Optional: 5-fold cross-validation
    if run_cv:
        print("="*80)
        print("5-FOLD CROSS-VALIDATION (Linear SVM)")
        cv_scores = cross_validate_svm(linear_results["pipeline"], X_train, y_train, cv=5)
        print("CV ROC-AUC scores:", cv_scores)
        print("Mean ROC-AUC:", cv_scores.mean())
        print("Std ROC-AUC:", cv_scores.std())

if __name__ == "__main__":
    # Reduce sample fraction to speed up runs for testing
    main(sample_fraction=0.3, run_cv=False)
