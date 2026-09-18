import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


DATA_PATH = "data/insurance.csv"
MODEL_PATH = "model/model.json"

FEATURE_ORDER = [
    "CreditScore",
    "Geography_Germany",
    "Geography_Spain",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary"
]

# Load and prepare dataset
def load_and_Prepare_data(path: str = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    dataset = pd.read_csv(DATA_PATH)

    # Handling categorical variables
    dataset.drop(['CustomerId', 'Surname'], axis=1, inplace=True)

    # One hot encode Geography
    geography_dummies = pd.get_dummies(
        dataset['Geography'],
        prefix="Geography",
        drop_first=True
    )
    for col in ["Geography_Germany", "Geography_Spain"]:
        if col not in geography_dummies.columns:
            geography_dummies[col] = 0
    dataset = pd.concat([
        dataset.drop("Geography", axis=1),
        geography_dummies
    ], axis=1)

    # Encode Gender Female -> 0, Male -> 1
    dataset['Gender'] = dataset['Gender'].apply(lambda x: 0 if x == 'Female' else 1)

    x = dataset[FEATURE_ORDER]
    y = dataset["Exited"]
    return x, y

def train_and_save_model() -> None:
    x, y = load_and_Prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0, stratify=y
    )

    # Build and train the model
    model = xgb.XGBClassifier(
        max_depth=8,
        learning_rate=0.3,
        n_estimators=100,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    os.makedirs("model", exist_ok=True)

    # Save Model
    model.save_model(MODEL_PATH)
    print(f"Model saved as '{MODEL_PATH}")


if __name__ == "__main__":
    train_and_save_model()