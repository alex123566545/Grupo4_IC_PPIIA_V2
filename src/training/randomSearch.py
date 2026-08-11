import os
import pickle

import numpy as np
import pandas as pd

from scipy.stats import randint

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score
)

from src.config.database import get_connection

from src.config.settings import (
    FEATURES,
    TARGET,
    RANDOM_STATE,
    MODEL_PATH,
    ENCODERS_PATH
)


RECALL_MINIMO = 0.60


def mejor_umbral(y_true, y_proba, recall_minimo=RECALL_MINIMO):

    candidatos = []

    for i in range(5, 100):

        umbral = i / 100

        y_pred = (
            y_proba >= umbral
        ).astype(int)

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
                (
                    umbral,
                    precision,
                    recall
                )
            )

    if not candidatos:

        resultados = []

        for i in range(5, 100):

            umbral = i / 100

            y_pred = (
                y_proba >= umbral
            ).astype(int)

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
                (
                    umbral,
                    precision,
                    recall
                )
            )

        return max(
            resultados,
            key=lambda x: x[2]
        )

    return max(
        candidatos,
        key=lambda x: x[1]
    )


def scoring_precision_recall(
    estimator,
    X,
    y
):

    y_proba = estimator.predict_proba(
        X
    )[:, 1]

    umbral, precision, recall = mejor_umbral(
        y,
        y_proba,
        RECALL_MINIMO
    )

    if recall < RECALL_MINIMO:

        return 0.0

    return precision


def cargar_datos():

    print("🔌 Conectando a Supabase...")

    conn = get_connection()

    try:

        print(
            "📥 Leyendo "
            "gold_ml.dataset_features..."
        )

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

        print(
            "🔌 Conexión cerrada."
        )

    filas_antes = len(df)

    df = df[
        df[TARGET].notna()
    ].copy()

    print(
        f"ℹ️ Filas excluidas por "
        f"target nulo: "
        f"{filas_antes - len(df)}"
    )

    print(
        f"ℹ️ Filas totales: "
        f"{len(df)}"
    )

    if "id_lote" not in df.columns:

        raise ValueError(
            "❌ No existe la columna "
            "'id_lote' en dataset_features."
        )

    print(
        f"ℹ️ Lotes totales: "
        f"{df['id_lote'].nunique()}"
    )

    return df


def preparar_datos(df):

    X = df[FEATURES].copy()

    y = df[TARGET].astype(int)

    groups = df["id_lote"].copy()

    encoders = {}

    categorical_columns = (
        X.select_dtypes(
            include=[
                "object",
                "category"
            ]
        ).columns
    )

    for col in categorical_columns:

        le = LabelEncoder()

        X[col] = le.fit_transform(
            X[col].astype(str)
        )

        encoders[col] = le

    return (
        X,
        y,
        groups,
        encoders
    )


def ejecutar_randomized_search(
    X,
    y,
    groups
):

    print("\n")
    print("=" * 70)
    print(
        "🔎 RANDOMIZED SEARCH + GROUPKFOLD"
    )
    print("=" * 70)

    group_kfold = GroupKFold(
        n_splits=5
    )

    modelo_base = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    param_distributions = {

        "n_estimators": randint(
            300,
            1001
        ),

        "max_depth": [
            None,
            6,
            8,
            10,
            12,
            15,
            20
        ],

        "min_samples_split": randint(
            2,
            15
        ),

        "min_samples_leaf": randint(
            1,
            10
        ),

        "max_features": [
            "sqrt",
            "log2",
            None
        ],

        "bootstrap": [
            True,
            False
        ],

        "class_weight": [
            "balanced",
            "balanced_subsample",
            None
        ],

        "criterion": [
            "gini",
            "entropy",
            "log_loss"
        ]
    }

    total_combinaciones = 100

    print(
        f"🔢 Combinaciones aleatorias: "
        f"{total_combinaciones}"
    )

    print(
        "📁 Validación: GroupKFold "
        "por id_lote"
    )

    print(
        "🎯 Objetivo: maximizar Precision "
        "manteniendo Recall >= 0.60"
    )

    search = RandomizedSearchCV(

        estimator=modelo_base,

        param_distributions=(
            param_distributions
        ),

        n_iter=total_combinaciones,

        scoring=(
            scoring_precision_recall
        ),

        cv=group_kfold,

        random_state=RANDOM_STATE,

        n_jobs=-1,

        verbose=1,

        refit=True
    )

    search.fit(
        X,
        y,
        groups=groups
    )

    print("\n")
    print("=" * 70)
    print(
        "🏆 RANDOMIZED SEARCH FINALIZADO"
    )
    print("=" * 70)

    print(
        "\nMejores hiperparámetros:"
    )

    for parametro, valor in (
        search.best_params_.items()
    ):

        print(
            f"   {parametro}: {valor}"
        )

    print(
        f"\nMejor Precision CV "
        f"con Recall >= 0.60: "
        f"{search.best_score_:.3f}"
    )

    return search


def evaluar_mejor_modelo(
    search,
    X,
    y,
    groups
):

    print("\n")
    print("=" * 70)
    print(
        "📊 EVALUACIÓN DEL MEJOR MODELO"
    )
    print("=" * 70)

    modelo = search.best_estimator_

    group_kfold = GroupKFold(
        n_splits=5
    )

    resultados = []

    for fold, (
        train_idx,
        test_idx
    ) in enumerate(
        group_kfold.split(
            X,
            y,
            groups
        ),
        start=1
    ):

        X_train = X.iloc[
            train_idx
        ]

        X_test = X.iloc[
            test_idx
        ]

        y_train = y.iloc[
            train_idx
        ]

        y_test = y.iloc[
            test_idx
        ]

        modelo_fold = (
            RandomForestClassifier(
                **search.best_params_,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )

        modelo_fold.fit(
            X_train,
            y_train
        )

        y_proba = (
            modelo_fold
            .predict_proba(X_test)[:, 1]
        )

        auc = roc_auc_score(
            y_test,
            y_proba
        )

        (
            umbral,
            precision,
            recall
        ) = mejor_umbral(
            y_test,
            y_proba
        )

        y_pred = (
            y_proba >= umbral
        ).astype(int)

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        resultados.append(
            {
                "auc": auc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "umbral": umbral
            }
        )

        print(
            f"\n📁 Fold {fold}"
        )

        print(
            f"   AUC:       {auc:.3f}"
        )

        print(
            f"   Precision: {precision:.3f}"
        )

        print(
            f"   Recall:    {recall:.3f}"
        )

        print(
            f"   F1:        {f1:.3f}"
        )

        print(
            f"   Umbral:    {umbral:.2f}"
        )

    auc_promedio = np.mean(
        [
            r["auc"]
            for r in resultados
        ]
    )

    precision_promedio = np.mean(
        [
            r["precision"]
            for r in resultados
        ]
    )

    recall_promedio = np.mean(
        [
            r["recall"]
            for r in resultados
        ]
    )

    f1_promedio = np.mean(
        [
            r["f1"]
            for r in resultados
        ]
    )

    umbral_promedio = np.mean(
        [
            r["umbral"]
            for r in resultados
        ]
    )

    print("\n")
    print("=" * 70)
    print(
        "📊 RESULTADOS PROMEDIO"
    )
    print("=" * 70)

    print(
        f"\nAUC-ROC promedio: "
        f"{auc_promedio:.3f}"
    )

    print(
        f"Precision promedio: "
        f"{precision_promedio:.3f}"
    )

    print(
        f"Recall promedio: "
        f"{recall_promedio:.3f}"
    )

    print(
        f"F1 promedio: "
        f"{f1_promedio:.3f}"
    )

    print(
        f"Umbral promedio: "
        f"{umbral_promedio:.2f}"
    )

    return modelo


def guardar_modelo(
    modelo,
    encoders
):

    os.makedirs(
        "models",
        exist_ok=True
    )

    with open(
        MODEL_PATH,
        "wb"
    ) as f:

        pickle.dump(
            modelo,
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

    print(
        "\n✅ Modelo guardado correctamente."
    )

    print(
        f"   Modelo: {MODEL_PATH}"
    )

    print(
        f"   Encoders: {ENCODERS_PATH}"
    )


def entrenar_modelo():

    df = cargar_datos()

    (
        X,
        y,
        groups,
        encoders
    ) = preparar_datos(df)

    search = ejecutar_randomized_search(
        X,
        y,
        groups
    )

    modelo_final = evaluar_mejor_modelo(
        search,
        X,
        y,
        groups
    )

    print("\n")
    print(
        "🚀 Entrenando modelo final "
        "con todo el dataset..."
    )

    modelo_final.fit(
        X,
        y
    )

    guardar_modelo(
        modelo_final,
        encoders
    )

    return modelo_final


if __name__ == "__main__":

    entrenar_modelo()
