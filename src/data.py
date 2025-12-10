from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split

def load_adult(verbose=False):
    adult = fetch_ucirepo(id=2)

    X = adult.data.features
    y = adult.data.targets

    if verbose:
        print("\n--- Metadata ---")
        print(adult.metadata)
        print("\n--- Variables ---")
        print(adult.variables)

    return X, y


def preprocess_adult(X, y, test_size=0.2, random_state=42):

    # Clean target to 0/1
    y = y.squeeze().map({
    '>50K': 1,
    '<=50K': 0,
    '>50K.': 1,  # Handle the variation with a trailing period
    '<=50K.': 0  # Handle the variation with a trailing period
    })

    # Identify column types
    cat_cols = X.select_dtypes(include=['object']).columns
    num_cols = X.select_dtypes(exclude=['object']).columns

    # Fill missing categorical values
    X[cat_cols] = X[cat_cols].fillna('Unknown')

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, cat_cols, num_cols

def main() -> None:
    X, y = load_adult()
    X_train, X_test, y_train, y_test, cat_cols, num_cols = preprocess_adult(X, y)
    print(y_train.value_counts(normalize=True), y_test.value_counts(normalize=True))


if __name__ == "__main__":
    main()
