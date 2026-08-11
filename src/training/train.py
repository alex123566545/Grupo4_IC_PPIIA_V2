import os
import pickle

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config.settings import (
    FEATURES,
    TARGET,
    TEST_SIZE,
    RANDOM_STATE,
    MODEL_PATH,
    ENCODERS_PATH
)


def train_model(conn):

    query = """
        SELECT *
        FROM gold_ml.dataset_features
    """

    df = pd.read_sql(query, conn)

    filas_antes = len(df)

    df = df[df[TARGET].notna()].copy()

    print(
        f"ℹ️ Filas excluidas por target nulo "
        f"(sin futuro suficiente): "
        f"{filas_antes - len(df)}"
    )

    X = df[FEATURES].copy()

    y = df[TARGET].astype(int)

    encoders = {}

    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns

    for col in categorical_columns:

        le = LabelEncoder()

        X[col] = le.fit_transform(
            X[col].astype(str)
        )

        encoders[col] = le

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    os.makedirs("models", exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump(encoders, f)

    print("✅ Modelo entrenado correctamente.")

    return model, X_test, y_test

