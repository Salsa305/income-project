"""
XGBoost Classification Models with Loss Function Experiments

This module implements XGBoost classifiers with various loss functions to explore
how different objectives affect model behavior on the income classification task.

Loss Functions Explored:
1. 'binary:logistic' - Standard binary cross-entropy (baseline)
2. 'binary:logitraw' - Logistic loss with raw predictions
3. 'binary:hinge' - Hinge loss (SVM-like)
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    average_precision_score,
    fbeta_score, # Import fbeta_score for F0.5
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from typing import Dict


def run_xgboost(
    X_train, X_test, y_train, y_test,
    preprocessor,
    objective="binary:logistic",
    max_depth=5,
    learning_rate=0.1,
    n_estimators=100,
    subsample=1.0,
    colsample_bytree=1.0,
    scale_pos_weight=None,
    early_stopping_rounds=None,
    eval_set=None,
) -> Dict:
    """
    Train and evaluate an XGBoost classifier.

    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test labels
        preprocessor: sklearn preprocessor (ColumnTransformer)
        objective: Loss function to use. Options: 'binary:logistic', etc.
        max_depth: Maximum tree depth
        learning_rate: Learning rate (eta)
        n_estimators: Number of boosting rounds
        subsample: Subsample ratio of training instances
        colsample_bytree: Subsample ratio of features
        scale_pos_weight: Weight for positive class
        early_stopping_rounds: Rounds for early stopping (requires eval_set)
        eval_set: Evaluation set for early stopping

    Returns:
        dict with keys:
            - 'pipeline': Fitted sklearn Pipeline
            - 'auc': ROC-AUC score on test set
            - 'pr_auc': PR-AUC score on test set (NEW)
            - 'f05_score': F0.5 score on test set (Precision-favored) (UPDATED)
            - 'report': Custom classification report (UPDATED)
            - 'y_pred': Predicted labels
            - 'y_proba': Predicted probabilities
            - 'confusion_matrix': Confusion matrix
    """

    model = XGBClassifier(
        objective=objective,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=early_stopping_rounds,
        eval_metric="auc",
        random_state=42,
        verbosity=0,
    )

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test) # Predictions at default threshold 0.5
    y_proba = pipe.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    
    # --- 1. Calculate Ranking Metrics ---
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    
    # --- 2. Calculate F0.5 Score at 0.5 Threshold ---
    # F0.5 score penalizes False Positives more than False Negatives
    f05_score_at_05 = fbeta_score(y_test, y_pred, beta=0.5, zero_division=0)

    # --- 3. Generate Classification Report (Full Standard Metrics) ---
    report_str = classification_report(y_test, y_pred, digits=4, output_dict=False)
    
    # --- 4. Append Goal-Aligned Metrics to the Report String ---
    summary_data = {
        'PR-AUC': [pr_auc],
        'ROC-AUC': [roc_auc]
    }
    df_summary = pd.DataFrame(summary_data, index=['Ranking Metrics']).T 
    summary_str = df_summary.to_string(float_format='%.4f')
    
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
        "auc": roc_auc, # ROC-AUC
        "pr_auc": pr_auc, # PR-AUC (New, important metric)
        "f05_score": f05_score_at_05, # F0.5 at default 0.5 threshold
        "report": final_report, # Customized report
        "y_pred": y_pred,
        "y_proba": y_proba,
        "confusion_matrix": cm,
    }


def cross_validate_xgboost(
    X,
    y,
    preprocessor,
    objective="binary:logistic",
    max_depth=5,
    learning_rate=0.1,
    n_estimators=100,
    cv=5,
    # CHANGE: Default scoring to PR-AUC (average_precision)
    scoring="average_precision",
):
    """
    Perform stratified k-fold cross-validation for XGBoost, defaulting to PR-AUC scoring.
    """

    model = XGBClassifier(
        objective=objective,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        random_state=42,
        verbosity=0,
    )

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=skf, scoring=scoring)

    return scores


def compare_objectives(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor,
    objectives=None,
    max_depth=5,
    learning_rate=0.1,
    n_estimators=100,
):
    """
    Compare multiple XGBoost objective functions.
    """
    # NOTE: run_xgboost handles the new reporting now.
    if objectives is None:
        objectives = ["binary:logistic", "binary:logitraw", "binary:hinge"]

    results = {}
    for obj in objectives:
        results[obj] = run_xgboost(
            X_train,
            X_test,
            y_train,
            y_test,
            preprocessor,
            objective=obj,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
        )
    return results


def hyperparameter_sweep(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor,
    objective="binary:logistic",
    max_depths=None,
    learning_rates=None,
    n_estimators_list=None,
    # NOTE: This sweep should be performed using PR-AUC as the sorting metric.
):
    """
    Perform a grid search over hyperparameters for XGBoost.
    """

    if max_depths is None:
        max_depths = [3, 5, 7, 9]
    if learning_rates is None:
        learning_rates = [0.01, 0.05, 0.1, 0.2]
    if n_estimators_list is None:
        n_estimators_list = [50, 100, 150, 200]

    results = {}

    for depth in max_depths:
        for lr in learning_rates:
            for n_est in n_estimators_list:
                key = (depth, lr, n_est)
                try:
                    # Use run_xgboost which calculates PR-AUC and returns it
                    result = run_xgboost(
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        preprocessor,
                        objective=objective,
                        max_depth=depth,
                        learning_rate=lr,
                        n_estimators=n_est,
                    )
                    
                    # CHANGE: Store PR-AUC instead of ROC-AUC
                    results[key] = {
                        "pr_auc": result["pr_auc"],
                        "report": result["report"],
                        "pipeline": result["pipeline"]
                    }
                except Exception as e:
                    print(f"Error with params {key}: {e}")
                    results[key] = None

    return results


def regularization_tuning_sweep(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor,
    objective="binary:logistic",
    max_depths=None,
    learning_rates=None,
    n_estimators_list=None,
    min_child_weights=None,
    subsamples=None,
    colsample_bytrees=None,
    gammas=None,
):
    """
    Tier 3: Regularization Tuning Sweep
    
    Performs grid search over regularization hyperparameters for XGBoost, 
    now tracking PR-AUC.
    """
    
    if max_depths is None:
        max_depths = [5, 7]
    if learning_rates is None:
        learning_rates = [0.1]
    if n_estimators_list is None:
        n_estimators_list = [150, 200]
    if min_child_weights is None:
        min_child_weights = [1, 3, 5]
    if subsamples is None:
        subsamples = [0.8, 1.0]
    if colsample_bytrees is None:
        colsample_bytrees = [0.8, 1.0]
    if gammas is None:
        gammas = [0, 0.1]

    results = {}
    total_combinations = (
        len(max_depths) * len(learning_rates) * len(n_estimators_list) *
        len(min_child_weights) * len(subsamples) * len(colsample_bytrees) * len(gammas)
    )
    
    print(f"Total combinations to test: {total_combinations}")
    count = 0

    for depth in max_depths:
        for lr in learning_rates:
            for n_est in n_estimators_list:
                for mcw in min_child_weights:
                    for ss in subsamples:
                        for csb in colsample_bytrees:
                            for gamma in gammas:
                                key = (depth, lr, n_est, mcw, ss, csb, gamma)
                                count += 1
                                try:
                                    model = XGBClassifier(
                                        objective=objective,
                                        max_depth=depth,
                                        learning_rate=lr,
                                        n_estimators=n_est,
                                        min_child_weight=mcw,
                                        subsample=ss,
                                        colsample_bytree=csb,
                                        gamma=gamma,
                                        random_state=42,
                                        verbosity=0,
                                    )

                                    pipe = Pipeline([
                                        ("preprocessor", preprocessor),
                                        ("model", model)
                                    ])

                                    pipe.fit(X_train, y_train)

                                    y_proba = pipe.predict_proba(X_test)[:, 1]
                                    # CHANGE: Score tracked is now PR-AUC
                                    pr_auc = average_precision_score(y_test, y_proba)
                                    
                                    results[key] = {
                                        "pr_auc": pr_auc,
                                        "pipeline": pipe,
                                    }
                                    
                                    if count % 10 == 0:
                                        print(f"  [{count}/{total_combinations}] PR-AUC: {pr_auc:.6f}")
                                        
                                except Exception as e:
                                    print(f"Error with params {key}: {e}")
                                    results[key] = None

    return results


def get_feature_importance(pipeline, top_n=20):
    """
    Extract feature importance from trained XGBoost model.
    (No changes needed here)
    """

    xgb_model = pipeline.named_steps["model"]
    importance = xgb_model.feature_importances_

    # Get feature names from preprocessor
    preprocessor = pipeline.named_steps["preprocessor"]
    
    # Get feature names after transformation
    feature_names = []
    
    # Add numeric features
    # Assuming preprocessor structure: [('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)]
    # Use preprocessor.transformers_[0][2] for num_cols and preprocessor.transformers_[1][2] for cat_cols
    
    # Handle numeric features (assuming they are first transformer)
    num_features = preprocessor.named_transformers_["num"].get_feature_names_out(
        preprocessor.transformers_[0][2]
    )
    feature_names.extend(num_features)
    
    # Handle categorical features (assuming they are second transformer)
    cat_features = preprocessor.named_transformers_["cat"].get_feature_names_out(
        preprocessor.transformers_[1][2]
    )
    feature_names.extend(cat_features)

    # Sort by importance
    sorted_idx = np.argsort(importance)[::-1][:top_n]
    
    importance_dict = {
        feature_names[i]: importance[i] for i in sorted_idx
    }

    return importance_dict


def engineer_features(X):
    """
    Create meaningful engineered features from raw data.
    (No changes needed here)
    """
    X_eng = X.copy()
    
    # 1. Replace capital gains/losses with binary flags + log-transformed magnitudes
    X_eng['has_capital_gain'] = (X_eng['capital-gain'] > 0).astype(int)
    X_eng['has_capital_loss'] = (X_eng['capital-loss'] > 0).astype(int)
    
    # Drop the raw zero-inflated features since we've captured their signal
    X_eng = X_eng.drop(['capital-gain', 'capital-loss'], axis=1)
    
    # 2. Interaction features
    X_eng['age_education_interaction'] = X_eng['age'] * X_eng['education-num']
    X_eng['hours_education_interaction'] = X_eng['hours-per-week'] * X_eng['education-num']
    
    # 3. Polynomial features
    X_eng['age_squared'] = X_eng['age'] ** 2
    
    # 4. Age groups (categorical binning for career lifecycle)
    X_eng['age_group'] = pd.cut(X_eng['age'], 
                                 bins=[0, 25, 35, 45, 55, 65, 100],
                                 labels=['<25', '25-35', '35-45', '45-55', '55-65', '65+'])
    
    return X_eng


def compare_feature_engineering(
    X_train_base, X_test_base, X_train_eng, X_test_eng, y_train, y_test,
    preprocessor_baseline,
    preprocessor_engineered,
    max_depth=5,
    learning_rate=0.1,
    n_estimators=100
):
    """
    Compare baseline features vs engineered features.
    (No changes needed here, as it calls the updated run_xgboost)
    """
    
    baseline_result = run_xgboost(
        X_train_base, X_test_base, y_train, y_test,
        preprocessor_baseline,
        objective="binary:logistic",
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators
    )
    
    engineered_result = run_xgboost(
        X_train_eng, X_test_eng, y_train, y_test,
        preprocessor_engineered,
        objective="binary:logistic",
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators
    )
    
    return {
        "baseline": baseline_result,
        "engineered": engineered_result
    }


def comprehensive_regularization_tuning(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor,
    objective="binary:logistic",
    max_depths=None,
    learning_rates=None,
    n_estimators_list=None,
    min_child_weights=None,
    subsamples=None,
    colsample_bytrees=None,
    gammas=None,
):
    """
    Comprehensive Regularization Tuning on Baseline Features
    
    Performs exhaustive grid search over regularization hyperparameters, 
    now tracking PR-AUC.
    """
    
    # Set defaults
    if max_depths is None:
        max_depths = [3, 5, 7, 9, 11]
    if learning_rates is None:
        learning_rates = [0.01, 0.05, 0.1, 0.15, 0.2]
    if n_estimators_list is None:
        n_estimators_list = [100, 150, 200, 250, 300]
    if min_child_weights is None:
        min_child_weights = [1, 3, 5, 7]
    if subsamples is None:
        subsamples = [0.7, 0.8, 0.9, 1.0]
    if colsample_bytrees is None:
        colsample_bytrees = [0.7, 0.8, 0.9, 1.0]
    if gammas is None:
        gammas = [0, 0.01, 0.1, 0.5]

    tuning_results = {}
    total_combinations = (
        len(max_depths) * len(learning_rates) * len(n_estimators_list) *
        len(min_child_weights) * len(subsamples) * len(colsample_bytrees) * len(gammas)
    )
    
    count = 0

    for depth in max_depths:
        for lr in learning_rates:
            for n_est in n_estimators_list:
                for mcw in min_child_weights:
                    for ss in subsamples:
                        for csb in colsample_bytrees:
                            for gamma in gammas:
                                count += 1
                                key = (depth, lr, n_est, mcw, ss, csb, gamma)
                                
                                model = XGBClassifier(
                                    objective=objective,
                                    max_depth=depth,
                                    learning_rate=lr,
                                    n_estimators=n_est,
                                    min_child_weight=mcw,
                                    subsample=ss,
                                    colsample_bytree=csb,
                                    gamma=gamma,
                                    random_state=42,
                                    verbosity=0,
                                )

                                pipe = Pipeline([
                                    ("preprocessor", preprocessor),
                                    ("model", model)
                                ])

                                pipe.fit(X_train, y_train)
                                y_proba = pipe.predict_proba(X_test)[:, 1]
                                # CHANGE: Score tracked is now PR-AUC
                                pr_auc_score = average_precision_score(y_test, y_proba)
                                
                                tuning_results[key] = pr_auc_score
                                
                                # Progress update every 64 combinations
                                if count % 64 == 0:
                                    print(f"  [{count:4d}/{total_combinations}] PR-AUC: {pr_auc_score:.6f}")

    # Get top 10 results
    top_results = sorted(tuning_results.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Format best parameters
    best_key, best_pr_auc = top_results[0]
    depth, lr, n_est, mcw, ss, csb, gamma = best_key
    
    best_params = {
        'max_depth': depth,
        'learning_rate': lr,
        'n_estimators': n_est,
        'min_child_weight': mcw,
        'subsample': ss,
        'colsample_bytree': csb,
        'gamma': gamma,
        'pr_auc': best_pr_auc, # Key changed from 'auc' to 'pr_auc'
    }

    return {
        'results': tuning_results,
        'top_results': top_results,
        'best_params': best_params,
        'total_combinations': total_combinations,
    }