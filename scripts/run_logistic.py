from src.data import load_adult, preprocess_adult
from src.preprocessing import make_preprocessor
from src.models.logistic import run_logistic

def main():
    X, y = load_adult()
    X_train, X_test, y_train, y_test, cat_cols, num_cols = preprocess_adult(X, y)

    preprocessor = make_preprocessor(cat_cols, num_cols)

    print("="*80)
    print("BASELINE LOGISTIC REGRESSION")
    print("="*80)

    baseline = run_logistic(
        X_train, X_test, y_train, y_test,
        preprocessor
    )

    print("ROC-AUC:", baseline["auc"])
    print(baseline["report"])

    print("="*80)
    print("L1 REGULARIZED LOGISTIC")
    print("="*80)

    l1 = run_logistic(
        X_train, X_test, y_train, y_test,
        preprocessor,
        penalty="l1",
        C=0.1
    )

    print("ROC-AUC:", l1["auc"])
    print(l1["report"])

if __name__ == "__main__":
    main()
