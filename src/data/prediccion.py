# ============================================================
# SIPREM-BOVINO
# PREDICCIÓN — RANDOM FOREST
#
# Fuente:
#     gold_ml.dataset_prediccion
#
# Archivos utilizados:
#     models/model.pkl
#     models/encoders.pkl
#     models/threshold.pkl
#
# Salida:
#     models/predicciones_dataset.csv
#
# IMPORTANTE:
#     dataset_prediccion NO debe contener el target.
#
# ============================================================


# ============================================================
# IMPORTACIONES
# ============================================================

import os
import pickle

import pandas as pd
import numpy as np

from src.config.settings import (
    FEATURES,
    MODEL_PATH,
    ENCODERS_PATH
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

PREDICTION_TABLE = (
    "gold_ml.dataset_prediccion"
)

OUTPUT_PATH = (
    "models/predicciones_dataset.csv"
)

THRESHOLD_PATH = (
    "models/threshold_rf_sin_lote29.pkl"
)


# ============================================================
# CARGAR MODELO
# ============================================================

def cargar_modelo():

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"No existe el modelo: "
            f"{MODEL_PATH}"
        )

    with open(
        MODEL_PATH,
        "rb"
    ) as f:

        model = pickle.load(f)

    return model


# ============================================================
# CARGAR ENCODERS
# ============================================================

def cargar_encoders():

    if not os.path.exists(ENCODERS_PATH):

        raise FileNotFoundError(
            f"No existen los encoders: "
            f"{ENCODERS_PATH}"
        )

    with open(
        ENCODERS_PATH,
        "rb"
    ) as f:

        encoders = pickle.load(f)

    return encoders


# ============================================================
# CARGAR UMBRAL
# ============================================================

def cargar_umbral():

    if not os.path.exists(
        THRESHOLD_PATH
    ):

        print(
            "⚠️ No se encontró threshold.pkl."
        )

        print(
            "⚠️ Se utilizará umbral = 0.50"
        )

        return 0.50

    with open(
        THRESHOLD_PATH,
        "rb"
    ) as f:

        threshold = pickle.load(f)

    return float(threshold)


# ============================================================
# TRANSFORMAR FEATURES
# ============================================================

def transformar_features(
    X,
    encoders
):

    X = X.copy()

    for col, encoder in encoders.items():

        if col not in X.columns:

            raise ValueError(
                f"La columna '{col}' "
                f"requerida por el encoder "
                f"no existe en "
                f"dataset_prediccion."
            )

        valores = (
            X[col]
            .astype(str)
        )

        mapping = {

            clase: i

            for i, clase in enumerate(
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

    # Igual que durante entrenamiento.
    if (
        "actividad_sensor_indice"
        in X.columns
    ):

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
# VALIDAR COLUMNAS
# ============================================================

def validar_features(df):

    faltantes = [
        col
        for col in FEATURES
        if col not in df.columns
    ]

    if faltantes:

        raise ValueError(
            "\n❌ Faltan columnas "
            "requeridas:\n"
            + "\n".join(
                f"   - {col}"
                for col in faltantes
            )
        )


# ============================================================
# PREDICCIÓN
# ============================================================

def realizar_prediccion():

    print(
        "\n" + "=" * 70
    )

    print(
        "🐄 SIPREM-BOVINO — PREDICCIÓN"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # CONEXIÓN
    # ========================================================

    print(
        "\n📥 Leyendo "
        "gold_ml.dataset_prediccion..."
    )

    from src.config.database import (
        get_connection
    )

    conn = get_connection()

    try:

        query = f"""
            SELECT *
            FROM {PREDICTION_TABLE}
            ORDER BY id_lote, fecha
        """

        df = pd.read_sql(
            query,
            conn
        )

    finally:

        conn.close()

    # ========================================================
    # VALIDACIÓN DATASET
    # ========================================================

    print(
        f"✓ Filas recibidas: "
        f"{len(df)}"
    )

    if df.empty:

        raise ValueError(
            "❌ dataset_prediccion "
            "está vacío."
        )

    # --------------------------------------------------------
    # Comprobar target
    # --------------------------------------------------------

    TARGET = (
        "target_riesgo_alto_4sem"
    )

    if TARGET in df.columns:

        raise ValueError(
            f"❌ dataset_prediccion "
            f"contiene la variable objetivo "
            f"'{TARGET}'.\n\n"
            "Este dataset debe contener "
            "únicamente las variables "
            "de entrada."
        )

    # --------------------------------------------------------
    # Comprobar features
    # --------------------------------------------------------

    validar_features(df)

    # ========================================================
    # MODELO
    # ========================================================

    print(
        "📦 Cargando modelo..."
    )

    model = cargar_modelo()

    print(
        "📦 Cargando encoders..."
    )

    encoders = cargar_encoders()

    print(
        "🎯 Cargando umbral..."
    )

    threshold = cargar_umbral()

    print(
        f"✓ Umbral utilizado: "
        f"{threshold:.2f}"
    )

    # ========================================================
    # METADATA
    # ========================================================

    metadata_columns = []

    if "id_lote" in df.columns:

        metadata_columns.append(
            "id_lote"
        )

    if "fecha" in df.columns:

        metadata_columns.append(
            "fecha"
        )

    metadata = (
        df[
            metadata_columns
        ]
        .copy()
    )

    # ========================================================
    # X
    # ========================================================

    X = (
        df[
            FEATURES
        ]
        .copy()
    )

    # ========================================================
    # NULOS
    # ========================================================

    X = preparar_nulos(
        X
    )

    # ========================================================
    # ENCODING
    # ========================================================

    X_encoded = (
        transformar_features(
            X,
            encoders
        )
    )

    # ========================================================
    # PREDICCIÓN
    # ========================================================

    print(
        "🤖 Ejecutando predicciones..."
    )

    y_proba = (
        model
        .predict_proba(
            X_encoded
        )[:, 1]
    )

    y_pred = (
        y_proba >= threshold
    ).astype(int)

    # ========================================================
    # RESULTADO
    # ========================================================

    resultado = metadata.copy()

    resultado[
        "probabilidad_riesgo_alto"
    ] = y_proba

    resultado[
        "prediccion_riesgo_alto"
    ] = y_pred

    resultado[
        "nivel_riesgo"
    ] = np.where(
        y_pred == 1,
        "ALTO",
        "BAJO"
    )

    resultado[
        "umbral_utilizado"
    ] = threshold

    # ========================================================
    # GUARDAR CSV
    # ========================================================

    os.makedirs(
        "models",
        exist_ok=True
    )

    resultado.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # RESUMEN
    # ========================================================

    positivos = int(
        y_pred.sum()
    )

    negativos = int(
        len(y_pred)
        - positivos
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "📊 RESULTADO DE PREDICCIÓN"
    )

    print(
        "=" * 70
    )

    print(
        f"Filas procesadas: "
        f"{len(resultado)}"
    )

    print(
        f"Riesgo ALTO: "
        f"{positivos}"
    )

    print(
        f"Riesgo BAJO: "
        f"{negativos}"
    )

    print(
        f"Umbral: "
        f"{threshold:.2f}"
    )

    print(
        f"\n📄 CSV generado:"
    )

    print(
        f"   {OUTPUT_PATH}"
    )

    print(
        "\n✅ PREDICCIÓN FINALIZADA"
    )

    print(
        "=" * 70
    )

    return resultado


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    realizar_prediccion()