from src.data import load_adult, preprocess_adult
from src.preprocessing import make_preprocessor
from src.models.xgboost import (
    run_xgboost,
    cross_validate_xgboost,
    compare_objectives,
    hyperparameter_sweep,
    get_feature_importance,
)
import numpy as np
import pandas as pd


def test_baseline(sample_fraction=1.0):
    """Test baseline XGBoost model with default settings."""
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    X, y = load_adult()

    # Optional: reduce dataset for faster experimentation
    if sample_fraction < 1.0:
        indices = np.random.RandomState(42).choice(
            len(X), size=int(len(X) * sample_fraction), replace=False
        )
        X = X.iloc[indices]
        y = y.iloc[indices]
        print(f"Using {len(X)} samples for quick experimentation")

    X_train, X_test, y_train, y_test, cat_cols, num_cols = preprocess_adult(X, y)
    preprocessor = make_preprocessor(cat_cols, num_cols)

    print(f"Train size: {X_train.shape}")
    print(f"Test size: {X_test.shape}")
    print(f"Class distribution (train): {y_train.value_counts().to_dict()}")
    print()

    # Baseline XGBoost
    print("=" * 80)
    print("BASELINE XGBOOST (binary:logistic)")
    print("=" * 80)

    baseline = run_xgboost(
        X_train, X_test, y_train, y_test, preprocessor, objective="binary:logistic"
    )

    print(f"ROC-AUC: {baseline['auc']:.4f}")
    print("\nClassification Report:")
    print(baseline["report"])
    print()

    return baseline, X_train, X_test, y_train, y_test, preprocessor, cat_cols, num_cols


def test_loss_functions(X_train, X_test, y_train, y_test, preprocessor):
    """Compare different loss functions."""
    print("=" * 80)
    print("COMPARING LOSS FUNCTIONS")
    print("=" * 80)

    objectives = ["binary:logistic", "binary:logitraw", "binary:hinge"]

    results = compare_objectives(
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        objectives=objectives,
        max_depth=5,
        learning_rate=0.1,
        n_estimators=100,
    )

    print("\nObjective Function Comparison:")
    print("-" * 80)

    for obj in objectives:
        print(f"\n{obj}:")
        print(f"  ROC-AUC: {results[obj]['auc']:.4f}")
        print(f"  Confusion Matrix:\n{results[obj]['confusion_matrix']}")

    # Create comparison dataframe
    comparison_df = pd.DataFrame(
        {
            "Objective": objectives,
            "ROC-AUC": [results[obj]["auc"] for obj in objectives],
        }
    )

    print("\n" + "=" * 80)
    print("SUMMARY TABLE:")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print()

    return results, comparison_df


def test_hyperparameter_sweep(X_train, X_test, y_train, y_test, preprocessor):
    """Perform hyperparameter sweep (limited for speed)."""
    print("=" * 80)
    print("HYPERPARAMETER SWEEP (Limited)")
    print("=" * 80)

    # Reduced parameter grid for quick testing
    max_depths = [3, 5, 7]
    learning_rates = [0.05, 0.1, 0.2]
    n_estimators_list = [50, 100]

    results = hyperparameter_sweep(
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        objective="binary:logistic",
        max_depths=max_depths,
        learning_rates=learning_rates,
        n_estimators_list=n_estimators_list,
    )

    # Extract results into dataframe
    results_list = []
    for (depth, lr, n_est), result in results.items():
        if result is not None:
            results_list.append(
                {
                    "max_depth": depth,
                    "learning_rate": lr,
                    "n_estimators": n_est,
                    "ROC-AUC": result["auc"],
                }
            )

    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values("ROC-AUC", ascending=False)

    print("\nTop 10 Hyperparameter Combinations:")
    print("=" * 80)
    print(results_df.head(10).to_string(index=False))
    print()

    return results_df


def test_cross_validation(X_train, y_train, preprocessor):
    """Perform 5-fold cross-validation."""
    print("=" * 80)
    print("5-FOLD CROSS-VALIDATION")
    print("=" * 80)

    cv_scores = cross_validate_xgboost(
        X_train,
        y_train,
        preprocessor,
        objective="binary:logistic",
        max_depth=5,
        learning_rate=0.1,
        n_estimators=100,
        cv=5,
        scoring="roc_auc",
    )

    print(f"CV ROC-AUC scores: {cv_scores}")
    print(f"Mean ROC-AUC: {cv_scores.mean():.4f}")
    print(f"Std ROC-AUC: {cv_scores.std():.4f}")
    print()

    return cv_scores


def test_class_weighting(X_train, X_test, y_train, y_test, preprocessor):
    """Test class weighting for imbalanced data."""
    print("=" * 80)
    print("CLASS WEIGHTING FOR IMBALANCED DATA")
    print("=" * 80)

    # Calculate scale_pos_weight based on class distribution
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count

    print(f"Negative class count: {neg_count}")
    print(f"Positive class count: {pos_count}")
    print(f"Computed scale_pos_weight: {scale_pos_weight:.4f}")
    print()

    # Baseline without weighting
    baseline = run_xgboost(
        X_train, X_test, y_train, y_test, preprocessor, objective="binary:logistic"
    )

    # With class weighting
    weighted = run_xgboost(
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        objective="binary:logistic",
        scale_pos_weight=scale_pos_weight,
    )

    print("Without Class Weighting:")
    print(f"  ROC-AUC: {baseline['auc']:.4f}")
    print(f"  Confusion Matrix:\n{baseline['confusion_matrix']}")

    print("\nWith Class Weighting (scale_pos_weight={:.4f}):".format(scale_pos_weight))
    print(f"  ROC-AUC: {weighted['auc']:.4f}")
    print(f"  Confusion Matrix:\n{weighted['confusion_matrix']}")
    print()

    return baseline, weighted, scale_pos_weight


def test_feature_importance(pipeline):
    """Extract and display feature importance."""
    print("=" * 80)
    print("TOP 15 MOST IMPORTANT FEATURES")
    print("=" * 80)

    importance_dict = get_feature_importance(pipeline, top_n=15)

    for i, (feature, importance) in enumerate(importance_dict.items(), 1):
        print(f"{i:2d}. {feature:40s} {importance:.6f}")

    print()


def main():
    """Run all tests."""
    print("\n")
    print("=" * 80)
    print("XGBOOST ANALYSIS FOR INCOME PREDICTION")
    print("=" * 80)
    print("\n")

    # Test with sample fraction for speed (change to 1.0 for full data)
    sample_frac = 0.3  # 30% of data for quick testing
    
    baseline, X_train, X_test, y_train, y_test, preprocessor, cat_cols, num_cols = test_baseline(
        sample_fraction=sample_frac
    )

    loss_results, comparison_df = test_loss_functions(
        X_train, X_test, y_train, y_test, preprocessor
    )

    hp_results = test_hyperparameter_sweep(
        X_train, X_test, y_train, y_test, preprocessor
    )

    cv_scores = test_cross_validation(X_train, y_train, preprocessor)

    baseline_unweighted, baseline_weighted, spw = test_class_weighting(
        X_train, X_test, y_train, y_test, preprocessor
    )

    test_feature_importance(baseline["pipeline"])

    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
