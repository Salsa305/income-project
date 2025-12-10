from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

def run_svm(
    X_train, X_test, y_train, y_test,
    preprocessor,
    kernel='linear',
    C=1.0,
    class_weight=None,
    probability=True
):
    """
    Train and evaluate an SVM model.
    
    Returns a dict with pipeline, AUC, and classification report.
    """
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
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    f05 = f1_score(y_test, y_pred, beta=0.5)

    return {
        "pipeline": pipe,
        "auc": roc_auc,
        "pr_auc": pr_auc,
        "f05_score": f05,
        "report": classification_report(y_test, y_pred)
    }


def cross_validate_svm(pipeline, X, y, cv=5, scoring='roc_auc'):
    """
    Perform stratified k-fold cross-validation for an SVM pipeline.
    Returns an array of scores.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=skf, scoring=scoring)
    return scores
