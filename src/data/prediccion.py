# ============================================================
# SIPREM-BOVINO
# PREDICCIÓN — RANDOM FOREST
# ============================================================
#
# Fuente:
#     gold_ml.dataset_prediccion
#
# Archivos utilizados:
#     models/model.pkl
#     models/encoders.pkl
#     models/threshold_rf_sin_lote29.pkl
#
# Salidas:
#     models/predicciones_dataset.csv
#     gold_ml.predicciones
#
# IMPORTANTE SOBRE EL SEGUIMIENTO:
#     Las columnas de intervención y verificación se dejan NULL/false
#     al momento de predecir. Se completan posteriormente cuando se
#     registra qué ocurrió después de la alerta.
#
# IMPORTANTE:
#     dataset_prediccion NO debe contener el target.
#
# El modelo utilizado corresponde al entrenamiento
# sin LOTE_29.
# ============================================================


# ============================================================
# IMPORTACIONES
# ============================================================

import os
import pickle

import numpy as np
import pandas as pd

from psycopg2.extras import execute_values

from src.config.settings import (
    FEATURES,
    MODEL_PATH,
    ENCODERS_PATH
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

PREDICTION_TABLE = "gold_ml.dataset_prediccion"

OUTPUT_PATH = "models/predicciones_dataset.csv"

# IMPORTANTE:
# Este es el umbral correspondiente al modelo sin LOTE_29.
THRESHOLD_PATH = "models/threshold_rf_sin_lote29.pkl"

# Identificador del modelo/version utilizado
MODELO_UTILIZADO = "random_forest_4sem_sin_lote29_v1"

TARGET = "target_riesgo_alto_4sem"


# ============================================================
# COLUMNAS QUE SE GUARDAN EN gold_ml.predicciones
# ============================================================

COLUMNAS_FEATURES_A_COPIAR = [
    "id_lote",
    "fecha",

    "distrito",
    "categoria_zootecnica",
    "raza_predominante",
    "altitud_msnm",
    "distancia_centro_veterinario_km",
    "tamano_lote_cabezas",

    "lote_sensorizado",
    "uso_registro_digital",

    "cobertura_vacunacion_pct",
    "dias_desde_desparasitacion",
    "animales_nuevos_30d",

    "casos_respiratorios",
    "casos_diarreicos",

    "temperatura_min_c",
    "temperatura_media_c",
    "temperatura_max_c",

    "humedad_relativa_pct",
    "precipitacion_semanal_mm",

    "condicion_pastura_indice",
    "indice_ndvi_satelital",

    "consumo_ms_kg_animal_dia",
    "agua_l_animal_dia",

    "actividad_sensor_indice",
    "condicion_corporal_prom",

    "precio_leche_local_s_kg",

    "semana_sin",
    "semana_cos",

    "media_movil_4s_pastura",
    "media_movil_4s_condicion_corporal",
    "media_movil_4s_temperatura",
]


# ============================================================
# CARGAR MODELO
# ============================================================

def cargar_modelo():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"\n❌ No existe el modelo:\n"
            f"   {MODEL_PATH}\n"
        )

    print(f"   ✓ Modelo: {MODEL_PATH}")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    return model


# ============================================================
# CARGAR ENCODERS
# ============================================================

def cargar_encoders():

    if not os.path.exists(ENCODERS_PATH):
        raise FileNotFoundError(
            f"\n❌ No existen los encoders:\n"
            f"   {ENCODERS_PATH}\n"
        )

    print(f"   ✓ Encoders: {ENCODERS_PATH}")

    with open(ENCODERS_PATH, "rb") as f:
        encoders = pickle.load(f)

    return encoders


# ============================================================
# CARGAR UMBRAL
# ============================================================

def cargar_umbral():

    if not os.path.exists(THRESHOLD_PATH):
        raise FileNotFoundError(
            f"\n❌ No existe el archivo de umbral:\n"
            f"   {THRESHOLD_PATH}\n\n"
            "No se utilizará un umbral por defecto.\n"
            "Esto evita generar predicciones con un umbral diferente\n"
            "al utilizado para este modelo."
        )

    print(f"   ✓ Umbral: {THRESHOLD_PATH}")

    with open(THRESHOLD_PATH, "rb") as f:
        threshold = pickle.load(f)

    threshold = float(threshold)

    if not 0 < threshold < 1:
        raise ValueError(
            f"\n❌ El umbral cargado no es válido: {threshold}\n"
            "Debe estar entre 0 y 1."
        )

    return threshold


# ============================================================
# TRANSFORMAR FEATURES CATEGÓRICAS
# ============================================================

def transformar_features(X, encoders):

    X = X.copy()

    for col, encoder in encoders.items():

        if col not in X.columns:
            raise ValueError(
                f"\n❌ La columna '{col}' requerida por el encoder "
                "no existe en dataset_prediccion."
            )

        # Convertimos a texto para mantener la misma lógica
        # utilizada durante el entrenamiento.
        valores = X[col].astype(str)

        mapping = {
            clase: i
            for i, clase in enumerate(encoder.classes_)
        }

        # Categorías desconocidas -> -1
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

    # Este NULL es estructural en lotes no sensorizados.
    if "actividad_sensor_indice" in X.columns:

        X["actividad_sensor_indice"] = (
            X["actividad_sensor_indice"]
            .fillna(-1)
        )

    return X


# ============================================================
# VALIDAR FEATURES
# ============================================================

def validar_features(df):

    faltantes = [
        col
        for col in FEATURES
        if col not in df.columns
    ]

    if faltantes:

        raise ValueError(
            "\n❌ Faltan columnas requeridas:\n"
            + "\n".join(
                f"   - {col}"
                for col in faltantes
            )
        )


# ============================================================
# VALIDAR IDENTIFICADORES
# ============================================================

def validar_identificadores(df):

    columnas_requeridas = [
        "id_lote",
        "fecha"
    ]

    faltantes = [
        col
        for col in columnas_requeridas
        if col not in df.columns
    ]

    if faltantes:

        raise ValueError(
            "\n❌ dataset_prediccion no contiene:\n"
            + "\n".join(
                f"   - {col}"
                for col in faltantes
            )
        )

    duplicados = df.duplicated(
        subset=["id_lote", "fecha"]
    )

    cantidad_duplicados = int(duplicados.sum())

    if cantidad_duplicados > 0:

        raise ValueError(
            f"\n❌ Se encontraron {cantidad_duplicados} "
            "filas duplicadas para la combinación "
            "(id_lote, fecha).\n\n"
            "Esto debe corregirse antes de generar "
            "las predicciones."
        )


# ============================================================
# GUARDAR EN gold_ml.predicciones
# ============================================================
# ============================================================
# CONVERTIR VALORES NUMPY -> PYTHON
# ============================================================

# ============================================================
# CONVERTIR VALORES NUMPY -> PYTHON
# ============================================================

def convertir_a_python(valor):

    if pd.isna(valor):
        return None

    if isinstance(valor, np.generic):
        return valor.item()

    return valor


# ============================================================
# GUARDAR EN gold_ml.predicciones
# ============================================================

def guardar_en_supabase(
    df_prediccion,
    y_proba,
    y_pred,
    umbral
):

    from src.config.database import get_connection

    columnas_insert = (
        COLUMNAS_FEATURES_A_COPIAR
        + [
            "modelo_utilizado",
            "umbral_utilizado",
            "probabilidad_riesgo_predicha",
            "riesgo_alto_predicho"
        ]
    )

    filas = []

    for i in range(len(df_prediccion)):

        fila_features = []

        for col in COLUMNAS_FEATURES_A_COPIAR:

            valor = df_prediccion.iloc[i][col]

            valor = convertir_a_python(valor)

            fila_features.append(valor)

        # ---------------------------------------------
        # CONVERSIONES CORRECTAS PARA POSTGRESQL
        # ---------------------------------------------

        probabilidad = float(y_proba[i])

        # IMPORTANTE:
        # PostgreSQL espera BOOLEAN, no INTEGER
        prediccion = bool(y_pred[i])

        umbral_python = float(umbral)

        fila = fila_features + [
            MODELO_UTILIZADO,
            umbral_python,
            probabilidad,
            prediccion
        ]

        # Conversión final por seguridad
        fila = tuple(
            convertir_a_python(valor)
            for valor in fila
        )

        filas.append(fila)

    columnas_sql = ", ".join(columnas_insert)

    placeholders = ", ".join(
        ["%s"] * len(columnas_insert)
    )

    query = f"""
        INSERT INTO gold_ml.predicciones (
            {columnas_sql}
        )
        VALUES %s

        ON CONFLICT (
            id_lote,
            fecha,
            modelo_utilizado
        )
        DO UPDATE SET

            probabilidad_riesgo_predicha =
                EXCLUDED.probabilidad_riesgo_predicha,

            riesgo_alto_predicho =
                EXCLUDED.riesgo_alto_predicho,

            umbral_utilizado =
                EXCLUDED.umbral_utilizado,

            predicho_en = now()

        -- IMPORTANTE:
        -- NO se actualizan aquí las columnas de seguimiento:
        --   intervencion_realizada
        --   tipo_intervencion
        --   fecha_intervencion
        --   resultado_intervencion
        --   verificado
        --   target_riesgo_alto_4sem_real
        --   fecha_verificacion
        --
        -- Si la predicción ya existía y posteriormente fue registrada
        -- una intervención o una verificación real, esos datos se
        -- conservan al volver a ejecutar este proceso.
    """

    conn = get_connection()

    try:

        with conn.cursor() as cur:

            execute_values(
                cur,
                query,
                filas,
                template=f"({placeholders})"
            )

        conn.commit()

        print(
            f"✓ {len(filas)} filas guardadas/actualizadas "
            "en gold_ml.predicciones"
        )

    except Exception as e:

        conn.rollback()

        print(
            "\n❌ Error guardando predicciones "
            "en Supabase:"
        )

        print(f"   {e}")

        raise

    finally:

        conn.close()
# ============================================================
# PREDICCIÓN
# ============================================================

def realizar_prediccion():

    print("\n")
    print("=" * 70)
    print("🐄 SIPREM-BOVINO — PREDICCIÓN RANDOM FOREST")
    print("=" * 70)

    # ========================================================
    # CONEXIÓN
    # ========================================================

    print("\n📥 Leyendo gold_ml.dataset_prediccion...")

    from src.config.database import get_connection

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
        f"✓ Filas recibidas: {len(df)}"
    )

    if df.empty:

        raise ValueError(
            "\n❌ gold_ml.dataset_prediccion está vacío."
        )

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    if TARGET in df.columns:

        raise ValueError(
            f"\n❌ dataset_prediccion contiene el target "
            f"'{TARGET}'.\n\n"
            "El dataset de predicción NO debe contener "
            "la variable objetivo."
        )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    validar_features(df)

    # --------------------------------------------------------
    # ID + FECHA
    # --------------------------------------------------------

    validar_identificadores(df)

    # ========================================================
    # CARGAR MODELO
    # ========================================================

    print("\n📦 Cargando archivos del modelo...")

    model = cargar_modelo()

    encoders = cargar_encoders()

    threshold = cargar_umbral()

    print(
        f"✓ Umbral utilizado: {threshold:.6f}"
    )

    # ========================================================
    # VALIDAR ORDEN DE FEATURES
    # ========================================================

    if hasattr(model, "n_features_in_"):

        cantidad_features_modelo = (
            model.n_features_in_
        )

        cantidad_features_actuales = len(FEATURES)

        if cantidad_features_modelo != cantidad_features_actuales:

            raise ValueError(
                "\n❌ La cantidad de features no coincide.\n"
                f"   Modelo: {cantidad_features_modelo}\n"
                f"   Configuración: {cantidad_features_actuales}\n\n"
                "Revisa settings.py y el modelo entrenado."
            )

    # ========================================================
    # METADATA
    # ========================================================

    metadata = df[
        ["id_lote", "fecha"]
    ].copy()

    # ========================================================
    # X
    # ========================================================

    X = df[FEATURES].copy()

    # Manejo de nulos
    X = preparar_nulos(X)

    # Aplicar mismos encoders del entrenamiento
    X_encoded = transformar_features(
        X,
        encoders
    )

    # ========================================================
    # VALIDACIÓN FINAL
    # ========================================================

    if list(X_encoded.columns) != list(FEATURES):

        raise ValueError(
            "\n❌ El orden de las columnas codificadas "
            "no coincide con FEATURES."
        )

    # ========================================================
    # PREDICCIÓN
    # ========================================================

    print("\n🤖 Ejecutando predicciones...")

    y_proba = model.predict_proba(
        X_encoded
    )[:, 1]

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
    # CREAR CARPETA
    # ========================================================

    os.makedirs(
        "models",
        exist_ok=True
    )

    # ========================================================
    # GUARDAR CSV
    # ========================================================

    resultado.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # GUARDAR SUPABASE
    # ========================================================

    print(
        "\n💾 Guardando predicciones "
        "en gold_ml.predicciones..."
    )

    guardar_en_supabase(
        df_prediccion=df,
        y_proba=y_proba,
        y_pred=y_pred,
        umbral=threshold
    )

    # ========================================================
    # RESUMEN
    # ========================================================

    positivos = int(
        y_pred.sum()
    )

    negativos = int(
        len(y_pred) - positivos
    )

    porcentaje_alto = (
        positivos / len(y_pred)
    ) * 100

    porcentaje_bajo = (
        negativos / len(y_pred)
    ) * 100

    print("\n")
    print("=" * 70)
    print("📊 RESULTADO DE PREDICCIÓN")
    print("=" * 70)

    print(
        f"Filas procesadas       : {len(resultado)}"
    )

    print(
        f"Riesgo ALTO            : {positivos}"
    )

    print(
        f"Riesgo BAJO            : {negativos}"
    )

    print(
        f"% Riesgo ALTO          : {porcentaje_alto:.2f}%"
    )

    print(
        f"% Riesgo BAJO          : {porcentaje_bajo:.2f}%"
    )

    print(
        f"Umbral utilizado       : {threshold:.6f}"
    )

    print(
        f"Modelo                 : {MODELO_UTILIZADO}"
    )

    print(
        f"\n📄 CSV generado:"
    )

    print(
        f"   {OUTPUT_PATH}"
    )

    print(
        "\n🗄️ Tabla actualizada:"
    )

    print(
        "   gold_ml.predicciones"
    )

    print("\n✅ PREDICCIÓN FINALIZADA")

    print("=" * 70)

    return resultado


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    realizar_prediccion()