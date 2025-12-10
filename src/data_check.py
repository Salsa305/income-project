from data import load_adult, preprocess_adult

def inspect_adult_data():
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    X, y = load_adult()
    X_train, X_test, y_train, y_test, cat_cols, num_cols = preprocess_adult(X, y)

    print("\n" + "=" * 80)
    print("TARGET DISTRIBUTION (y_train)")
    print("=" * 80)

    print("\nCounts:")
    print(y_train.value_counts())

    print("\nProportions:")
    print(y_train.value_counts(normalize=True))

    print("\n" + "=" * 80)
    print("COLUMN SPLITS")
    print("=" * 80)

    print("\nCategorical columns:")
    print(list(cat_cols))

    print("\nNumeric columns:")
    print(list(num_cols))

    print("\n" + "=" * 80)
    print("MISSING VALUES CHECK (TRAIN SET)")
    print("=" * 80)

    print("\nCategorical missing values:")
    print(X_train[cat_cols].isna().sum())

    print("\nNumeric missing values:")
    print(X_train[num_cols].isna().sum())

    print("\n" + "=" * 80)
    print("CATEGORICAL CARDINALITY")
    print("=" * 80)

    print(X_train[cat_cols].nunique().sort_values(ascending=False))

    print("\n" + "=" * 80)
    print("NUMERIC FEATURE DISTRIBUTION")
    print("=" * 80)

    print(X_train[num_cols].describe().T)

    print("\n" + "=" * 80)
    print("SAMPLE ROWS")
    print("=" * 80)

    print("\nX_train head:")
    print(X_train.head())

    print("\ny_train head:")
    print(y_train.head())


if __name__ == "__main__":
    inspect_adult_data()
