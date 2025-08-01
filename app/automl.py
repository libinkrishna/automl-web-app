from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def run_automl(df, target_column):
    df = df.copy()

    # Drop ID or non-informative columns if they exist
    if "id" in df.columns:
        df.drop("id", axis=1, inplace=True)

    # Encode categorical target if necessary
    if df[target_column].dtype == "object":
        le = LabelEncoder()
        df[target_column] = le.fit_transform(df[target_column])

    # Encode categorical features
    for col in df.select_dtypes(include=["object"]).columns:
        if col != target_column:
            df[col] = LabelEncoder().fit_transform(df[col])

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    tpot = TPOTClassifier(verbosity=2, generations=5, population_size=20, n_jobs=-1)
    tpot.fit(X_train, y_train)

    report = tpot.score(X_test, y_test)
    predictions = pd.DataFrame({
        "Actual": y_test,
        "Predicted": tpot.predict(X_test)
    })

    return tpot.fitted_pipeline_, f"Accuracy: {report:.2f}", predictions
