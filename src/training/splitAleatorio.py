# ============================================================
# SIPREM-BOVINO
# RANDOM FOREST — VALIDACIÓN GROUPKFOLD
#
# Correcciones metodológicas:
#
# 1. El umbral NO se selecciona usando el conjunto TEST.
# 2. El umbral se obtiene mediante predicciones OOF
#    de una validación interna agrupada por lote.
# 3. Los LabelEncoder se ajustan únicamente con TRAIN.
# 4. El TEST externo queda completamente aislado.
#
# ============================================================

import os
import pickle

import pandas as pd
import numpy as np

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


# ============================================================
# CONFIGURACIÓN
# ============================================================

N_SPLITS_EXTERNOS = 5

N_SPLITS_INTERNOS = 4

RECALL_MINIMO = 0.60


# ============================================================
# BUSCAR MEJOR UMBRAL
# ============================================================

def mejor_umbral(
    y_true,
    y_proba,
    recall_minimo=0.60
):
    """
    Busca el umbral que maximiza Precision
    manteniendo Recall >= recall_minimo.

    IMPORTANTE:
    Esta función solamente debe utilizarse
    sobre datos de validación, nunca sobre TEST.
    """

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

    # --------------------------------------------------------
    # Si encontramos candidatos que cumplen Recall >= 0.60
    # --------------------------------------------------------

    if candidatos:

        return max(
            candidatos,
            key=lambda x: x[1]
        )

    # --------------------------------------------------------
    # Si ningún umbral cumple el Recall mínimo,
    # seleccionamos el que maximiza Recall.
    # --------------------------------------------------------

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


# ============================================================
# PREPARACIÓN DE FEATURES
# ============================================================

def ajustar_encoders(X_train):
    """
    Ajusta los LabelEncoder exclusivamente con TRAIN.

    Devuelve:
        X_train_transformado
        encoders
    """

    X_train = X_train.copy()

    encoders = {}

    categorical_columns = X_train.select_dtypes(
        include=["object", "category"]
    ).columns

    for col in categorical_columns:

        encoder = LabelEncoder()

        valores = (
            X_train[col]
            .astype(str)
        )

        encoder.fit(valores)

        X_train[col] = encoder.transform(
            valores
        )

        encoders[col] = encoder

    return X_train, encoders


def transformar_features(
    X,
    encoders
):
    """
    Transforma un dataset utilizando
    encoders previamente ajustados.

    Las categorías desconocidas reciben -1.
    """

    X = X.copy()

    for col, encoder in encoders.items():

        valores = (
            X[col]
            .astype(str)
        )

        mapping = {
            clase: i
            for i, clase
            in enumerate(
                encoder.classes_
            )
        }

        X[col] = (
            valores
            .map(mapping)
            .fillna(-1)
            .astype(int)
        )

    return X


# ============================================================
# MANEJO DE NULOS
# ============================================================

def preparar_nulos(X):

    X = X.copy()

    if "actividad_sensor_indice" in X.columns:

        nulos_antes = (
            X["actividad_sensor_indice"]
            .isna()
            .sum()
        )

        X[
            "actividad_sensor_indice"
        ] = (
            X[
                "actividad_sensor_indice"
            ]
            .fillna(-1)
        )

    return X


# ============================================================
# CREAR RANDOM FOREST
# ============================================================

def crear_modelo():

    return RandomForestClassifier(

        n_estimators=300,

        max_depth=8,

        min_samples_split=5,

        min_samples_leaf=5,

        class_weight="balanced",

        random_state=RANDOM_STATE,

        n_jobs=-1
    )


# ============================================================
# SELECCIÓN DEL UMBRAL MEDIANTE OOF
# ============================================================

def obtener_umbral_oof(
    X_train,
    y_train,
    groups_train
):
    """
    Selecciona el umbral utilizando únicamente
    el TRAIN externo.

    Se realiza una validación interna GroupKFold.

    Las probabilidades OOF se generan de forma que
    cada registro es predicho por un modelo que NO
    lo utilizó para entrenar.

    Esto evita utilizar el TEST externo para seleccionar
    el umbral.
    """

    print(
        "\n   🔎 Seleccionando umbral "
        "mediante validación interna..."
    )

    inner_kfold = GroupKFold(
        n_splits=N_SPLITS_INTERNOS
    )

    y_oof = np.zeros(
        len(y_train),
        dtype=float
    )

    for inner_fold, (
        inner_train_idx,
        inner_valid_idx
    ) in enumerate(
        inner_kfold.split(
            X_train,
            y_train,
            groups_train
        ),
        start=1
    ):

        print(
            f"      Inner Fold {inner_fold}/"
            f"{N_SPLITS_INTERNOS}"
        )

        X_inner_train = X_train.iloc[
            inner_train_idx
        ].copy()

        X_inner_valid = X_train.iloc[
            inner_valid_idx
        ].copy()

        y_inner_train = y_train.iloc[
            inner_train_idx
        ]

        groups_inner_train = groups_train.iloc[
            inner_train_idx
        ]

        # ----------------------------------------------------
        # Nulos
        # ----------------------------------------------------

        X_inner_train = preparar_nulos(
            X_inner_train
        )

        X_inner_valid = preparar_nulos(
            X_inner_valid
        )

        # ----------------------------------------------------
        # Encoders SOLO con inner train
        # ----------------------------------------------------

        (
            X_inner_train,
            inner_encoders
        ) = ajustar_encoders(
            X_inner_train
        )

        X_inner_valid = transformar_features(
            X_inner_valid,
            inner_encoders
        )

        # ----------------------------------------------------
        # Modelo interno
        # ----------------------------------------------------

        model = crear_modelo()

        model.fit(
            X_inner_train,
            y_inner_train
        )

        proba = model.predict_proba(
            X_inner_valid
        )[:, 1]

        y_oof[
            inner_valid_idx
        ] = proba

    # ========================================================
    # Ahora sí buscamos el umbral.
    #
    # IMPORTANTE:
    # y_oof pertenece exclusivamente al TRAIN externo.
    # El TEST externo todavía NO fue utilizado.
    # ========================================================

    umbral, precision, recall = mejor_umbral(
        y_train,
        y_oof,
        RECALL_MINIMO
    )

    f1 = f1_score(
        y_train,
        (
            y_oof >= umbral
        ).astype(int),
        zero_division=0
    )

    print(
        f"      Umbral seleccionado: "
        f"{umbral:.2f}"
    )

    print(
        f"      Precision OOF: "
        f"{precision:.3f}"
    )

    print(
        f"      Recall OOF: "
        f"{recall:.3f}"
    )

    print(
        f"      F1 OOF: "
        f"{f1:.3f}"
    )

    return umbral


# ============================================================
# EVALUAR TEST EXTERNO
# ============================================================

def evaluar_test(
    y_test,
    y_proba,
    umbral
):
    """
    Evalúa el TEST externo utilizando un umbral
    previamente seleccionado con TRAIN.

    Aquí ya NO se busca ningún umbral.
    """

    auc = roc_auc_score(
        y_test,
        y_proba
    )

    y_pred = (
        y_proba >= umbral
    ).astype(int)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y_test,
            y_pred
        ).ravel()
    )

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


# ============================================================
# ENTRENAMIENTO PRINCIPAL
# ============================================================

def entrenar_modelo():

    print(
        "🔌 Conectando a Supabase..."
    )

    from src.config.database import (
        get_connection
    )

    conn = get_connection()

    try:

        print(
            "📥 Leyendo "
            "gold_ml.dataset_features..."
        )

        query = """
            SELECT *
            FROM gold_ml.dataset_features
            ORDER BY id_lote, fecha
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

    # ========================================================
    # FILTRAR TARGET
    # ========================================================

    filas_antes = len(df)

    df = df[
        df[TARGET].notna()
    ].copy()

    print(
        f"\nℹ️ Filas excluidas por "
        f"target nulo: "
        f"{filas_antes - len(df)}"
    )

    print(
        f"ℹ️ Filas totales: "
        f"{len(df)}"
    )

    # ========================================================
    # VERIFICAR LOTES
    # ========================================================

    if "id_lote" not in df.columns:

        raise ValueError(
            "❌ La columna 'id_lote' "
            "no existe."
        )

    groups = df[
        "id_lote"
    ].copy()

    print(
        f"ℹ️ Lotes totales: "
        f"{groups.nunique()}"
    )

    print(
        "\nℹ️ Distribución de registros "
        "por lote:"
    )

    print(
        groups.value_counts()
        .sort_index()
    )

    # ========================================================
    # FEATURES Y TARGET
    # ========================================================

    X = df[
        FEATURES
    ].copy()

    y = df[
        TARGET
    ].astype(int)

    # ========================================================
    # NULOS ESTRUCTURALES
    # ========================================================

    X = preparar_nulos(
        X
    )

    # ========================================================
    # GROUPKFOLD EXTERNO
    # ========================================================

    group_kfold = GroupKFold(
        n_splits=N_SPLITS_EXTERNOS
    )

    resultados = []

    print("\n")
    print("=" * 70)
    print(
        "🚀 GROUPKFOLD — "
        "VALIDACIÓN POR LOTE"
    )
    print("=" * 70)

    # ========================================================
    # FOLDS EXTERNOS
    # ========================================================

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

        print("\n")
        print("-" * 70)
        print(
            f"📁 FOLD {fold}"
        )
        print("-" * 70)

        X_train = X.iloc[
            train_idx
        ].copy()

        X_test = X.iloc[
            test_idx
        ].copy()

        y_train = y.iloc[
            train_idx
        ].copy()

        y_test = y.iloc[
            test_idx
        ].copy()

        groups_train = groups.iloc[
            train_idx
        ].copy()

        groups_test = groups.iloc[
            test_idx
        ].copy()

        print(
            f"   Lotes entrenamiento: "
            f"{groups_train.nunique()}"
        )

        print(
            f"   Lotes validación:    "
            f"{groups_test.nunique()}"
        )

        print(
            f"   Lotes TEST:"
        )

        print(
            f"      {sorted(groups_test.unique())}"
        )

        print(
            f"   Registros train: "
            f"{len(X_train)}"
        )

        print(
            f"   Registros test:  "
            f"{len(X_test)}"
        )

        # ====================================================
        # SELECCIONAR UMBRAL
        #
        # SOLO UTILIZA TRAIN EXTERNO
        # ====================================================

        umbral = obtener_umbral_oof(
            X_train,
            y_train,
            groups_train
        )

        # ====================================================
        # AJUSTAR ENCODERS SOLO CON TRAIN EXTERNO
        # ====================================================

        (
            X_train_encoded,
            encoders_fold
        ) = ajustar_encoders(
            X_train
        )

        X_test_encoded = transformar_features(
            X_test,
            encoders_fold
        )

        # ====================================================
        # ENTRENAR MODELO FINAL DEL FOLD
        #
        # Utiliza TODO el TRAIN externo.
        # ====================================================

        model = crear_modelo()

        model.fit(
            X_train_encoded,
            y_train
        )

        # ====================================================
        # PREDICCIÓN SOBRE TEST EXTERNO
        # ====================================================

        y_proba = model.predict_proba(
            X_test_encoded
        )[:, 1]

        # ====================================================
        # EVALUACIÓN
        #
        # AQUÍ NO SE TOCA EL UMBRAL.
        # ====================================================

        resultado = evaluar_test(
            y_test,
            y_proba,
            umbral
        )

        resultados.append(
            resultado
        )

        # ====================================================
        # RESULTADOS DEL FOLD
        # ====================================================

        print("\n   📊 RESULTADOS TEST")

        print(
            f"   AUC-ROC:   "
            f"{resultado['auc']:.3f}"
        )

        print(
            f"   Umbral:    "
            f"{resultado['umbral']:.2f}"
        )

        print(
            f"   Precision: "
            f"{resultado['precision']:.3f}"
        )

        print(
            f"   Recall:    "
            f"{resultado['recall']:.3f}"
        )

        print(
            f"   F1:        "
            f"{resultado['f1']:.3f}"
        )

        print(
            f"   TP={resultado['tp']} "
            f"FP={resultado['fp']} "
            f"FN={resultado['fn']} "
            f"TN={resultado['tn']}"
        )

    # ========================================================
    # RESULTADOS PROMEDIO
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "📊 RESULTADOS PROMEDIO "
        "GROUPKFOLD"
    )
    print("=" * 70)

    auc_promedio = np.mean([
        r["auc"]
        for r in resultados
    ])

    precision_promedio = np.mean([
        r["precision"]
        for r in resultados
    ])

    recall_promedio = np.mean([
        r["recall"]
        for r in resultados
    ])

    f1_promedio = np.mean([
        r["f1"]
        for r in resultados
    ])

    umbral_promedio = np.mean([
        r["umbral"]
        for r in resultados
    ])

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

    # ========================================================
    # RESULTADOS POR FOLD
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "📋 DETALLE DE FOLDS"
    )
    print("=" * 70)

    print()

    print(
        f"{'Fold':<7}"
        f"{'AUC':>8}"
        f"{'Umbral':>10}"
        f"{'Prec.':>10}"
        f"{'Recall':>10}"
        f"{'F1':>10}"
    )

    print(
        "-" * 55
    )

    for i, resultado in enumerate(
        resultados,
        start=1
    ):

        print(
            f"{i:<7}"
            f"{resultado['auc']:>8.3f}"
            f"{resultado['umbral']:>10.2f}"
            f"{resultado['precision']:>10.3f}"
            f"{resultado['recall']:>10.3f}"
            f"{resultado['f1']:>10.3f}"
        )

    # ========================================================
    # ACLARACIÓN METODOLÓGICA
    # ========================================================

    print("\n")
    print("=" * 70)

    print(
        "⚠️ EVALUACIÓN METODOLÓGICA"
    )

    print("=" * 70)

    print(
        "\nEl umbral de cada fold fue seleccionado "
        "utilizando únicamente el conjunto de "
        "entrenamiento mediante validación interna "
        "GroupKFold."
    )

    print(
        "\nLos lotes del TEST externo no participaron "
        "en la selección del umbral."
    )

    print(
        "\nPor tanto, Precision, Recall y F1 del "
        "TEST externo representan una evaluación "
        "más rigurosa."
    )

    # ========================================================
    # ENTRENAMIENTO FINAL
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "🚀 ENTRENANDO MODELO FINAL"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Ajustar encoder con TODO el dataset
    # --------------------------------------------------------

    (
        X_final,
        encoders_final
    ) = ajustar_encoders(
        X
    )

    modelo_final = crear_modelo()

    modelo_final.fit(
        X_final,
        y
    )

    # ========================================================
    # GUARDAR MODELO
    # ========================================================

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
            encoders_final,
            f
        )

    print(
        "\n✅ Modelo final entrenado "
        "correctamente."
    )

    print(
        f"📦 Modelo: "
        f"{MODEL_PATH}"
    )

    print(
        f"📦 Encoders: "
        f"{ENCODERS_PATH}"
    )

    print("\n")
    print("=" * 70)
    print(
        "✅ ENTRENAMIENTO FINALIZADO"
    )
    print("=" * 70)

    return modelo_final


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":

    entrenar_modelo()