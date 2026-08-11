
import os
import pickle

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.config.settings import (
    FEATURES,
    TARGET,
    RANDOM_STATE,
    MODEL_PATH,
    ENCODERS_PATH
)


def mejor_umbral(y_true, y_proba, recall_minimo=0.60):

    candidatos = []

    for i in range(5, 100):

        umbral = i / 100

        y_pred = (y_proba >= umbral).astype(int)

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0
        )

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0
        )

        if recall >= recall_minimo:

            candidatos.append(
                (umbral, precision, recall)
            )

    if not candidatos:

        resultados = []

        for i in range(5, 100):

            umbral = i / 100

            y_pred = (y_proba >= umbral).astype(int)

            recall = recall_score(
                y_true,
                y_pred,
                zero_division=0
            )

            precision = precision_score(
                y_true,
                y_pred,
                zero_division=0
            )

            resultados.append(
                (umbral, precision, recall)
            )

        umbral, precision, recall = max(
            resultados,
            key=lambda x: x[2]
        )

        return umbral, precision, recall

    return max(
        candidatos,
        key=lambda x: x[1]
    )


def evaluar_fold(y_true, y_proba):

    auc = roc_auc_score(
        y_true,
        y_proba
    )

    umbral, precision, recall = mejor_umbral(
        y_true,
        y_proba,
        recall_minimo=0.60
    )

    y_pred = (y_proba >= umbral).astype(int)

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred
    ).ravel()

    return {
        "auc": auc,
        "umbral": umbral,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }


def entrenar_modelo():

    print("🔌 Conectando a Supabase...")

    from src.config.database import get_connection

    conn = get_connection()

    try:

        print("📥 Leyendo gold_ml.dataset_features...")

        query = """
            SELECT *
            FROM gold_ml.dataset_features
        """

        df = pd.read_sql(
            query,
            conn
        )

    finally:

        conn.close()

        print("🔌 Conexión cerrada.")

    filas_antes = len(df)

    df = df[
        df[TARGET].notna()
    ].copy()

    print(
        f"ℹ️ Filas excluidas por target nulo: "
        f"{filas_antes - len(df)}"
    )

    print(
        f"ℹ️ Filas totales para GroupKFold: "
        f"{len(df)}"
    )

    if "id_lote" not in df.columns:

        raise ValueError(
            "❌ La columna 'id_lote' no existe en "
            "gold_ml.dataset_features."
        )

    groups = df["id_lote"].copy()

    print(
        f"ℹ️ Lotes totales: "
        f"{groups.nunique()}"
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

    model_params = {
        "n_estimators": 300,
        "max_depth": 8,
        "min_samples_split": 5,
        "min_samples_leaf": 5,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1
    }

    group_kfold = GroupKFold(
        n_splits=5
    )

    resultados = []

    print("\n")
    print("=" * 70)
    print("🚀 GROUPKFOLD — VALIDACIÓN POR LOTE")
    print("=" * 70)

    for fold, (train_idx, test_idx) in enumerate(
        group_kfold.split(
            X,
            y,
            groups
        ),
        start=1
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        grupos_train = groups.iloc[train_idx]
        grupos_test = groups.iloc[test_idx]

        print("\n" + "-" * 70)
        print(f"📁 FOLD {fold}")
        print("-" * 70)

        print(
            f"   Lotes entrenamiento: "
            f"{grupos_train.nunique()}"
        )

        print(
            f"   Lotes validación:    "
            f"{grupos_test.nunique()}"
        )

        print(
            f"   Registros train:     "
            f"{len(X_train)}"
        )

        print(
            f"   Registros test:      "
            f"{len(X_test)}"
        )

        model = RandomForestClassifier(
            **model_params
        )

        model.fit(
            X_train,
            y_train
        )

        y_proba = model.predict_proba(
            X_test
        )[:, 1]

        resultado = evaluar_fold(
            y_test,
            y_proba
        )

        resultados.append(
            resultado
        )

        print(
            f"   AUC-ROC:   "
            f"{resultado['auc']:.3f}"
        )

        print(
            f"   Umbral:     "
            f"{resultado['umbral']:.2f}"
        )

        print(
            f"   Precision:  "
            f"{resultado['precision']:.3f}"
        )

        print(
            f"   Recall:     "
            f"{resultado['recall']:.3f}"
        )

        print(
            f"   F1:         "
            f"{resultado['f1']:.3f}"
        )

        print(
            f"   TP={resultado['tp']} "
            f"FP={resultado['fp']} "
            f"FN={resultado['fn']} "
            f"TN={resultado['tn']}"
        )

    print("\n")
    print("=" * 70)
    print("📊 RESULTADOS PROMEDIO GROUPKFOLD")
    print("=" * 70)

    auc_promedio = sum(
        r["auc"] for r in resultados
    ) / len(resultados)

    precision_promedio = sum(
        r["precision"] for r in resultados
    ) / len(resultados)

    recall_promedio = sum(
        r["recall"] for r in resultados
    ) / len(resultados)

    f1_promedio = sum(
        r["f1"] for r in resultados
    ) / len(resultados)

    umbral_promedio = sum(
        r["umbral"] for r in resultados
    ) / len(resultados)

    print(
        f"\n   AUC-ROC promedio:  "
        f"{auc_promedio:.3f}"
    )

    print(
        f"   Precision promedio: "
        f"{precision_promedio:.3f}"
    )

    print(
        f"   Recall promedio:    "
        f"{recall_promedio:.3f}"
    )

    print(
        f"   F1 promedio:        "
        f"{f1_promedio:.3f}"
    )

    print(
        f"   Umbral promedio:    "
        f"{umbral_promedio:.2f}"
    )

    print("\n" + "=" * 70)

    print(
        "⚠️ Estos resultados representan una evaluación "
        "más rigurosa porque los lotes de validación "
        "no aparecen durante el entrenamiento."
    )

    print("=" * 70)

    print("\n🚀 Entrenando modelo final con todo el dataset...")

    modelo_final = RandomForestClassifier(
        **model_params
    )

    modelo_final.fit(
        X,
        y
    )

    os.makedirs(
        "models",
        exist_ok=True
    )

    with open(
        MODEL_PATH,
        "wb"
    ) as f:

        pickle.dump(
            modelo_final,
            f
        )

    with open(
        ENCODERS_PATH,
        "wb"
    ) as f:

        pickle.dump(
            encoders,
            f
        )

    print("✅ Modelo final entrenado correctamente.")

    return modelo_final


if __name__ == "__main__":

    entrenar_modelo()
