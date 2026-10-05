# ============================================================
# SIPREM-BOVINO
# RANDOM FOREST — TARGET DE 4 SEMANAS
#
# REENTRENAMIENTO CON DATOS DE PRODUCCIÓN
#
# Flujo:
#
#   1. Cargar histórico original.
#   2. Cargar predicciones de producción verificadas
#      y sin intervención.
#   3. Separar una parte de producción como HOLDOUT.
#   4. Entrenar candidato con:
#          histórico + producción de entrenamiento
#   5. Evaluar modelo actual y candidato sobre el MISMO
#      holdout de producción.
#   6. Guardar candidato en:
#          models/candidate/
#   7. Promover solamente si cumple los criterios.
#
# Estructura:
#
# models/
# ├── model.pkl
# ├── encoders.pkl
# ├── threshold.pkl
# │
# └── candidate/
#     ├── model.pkl
#     ├── encoders.pkl
#     └── threshold.pkl
#
# ============================================================


# ============================================================
# IMPORTACIONES
# ============================================================

import os
import pickle
import shutil

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
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
N_REPEATS = 5

RECALL_MINIMO = 0.60

# ------------------------------------------------------------
# Tolerancias para promoción
# ------------------------------------------------------------

# El candidato no puede perder más de este valor de Recall
# respecto al modelo actual.
TOLERANCIA_RECALL = 0.02

# El candidato no puede perder más de este valor de AUC
# respecto al modelo actual.
TOLERANCIA_AUC = 0.01

# ------------------------------------------------------------
# Lote excluido
# ------------------------------------------------------------

LOTE_EXCLUIDO = "LOTE_29"

# ------------------------------------------------------------
# Holdout de producción
# ------------------------------------------------------------

# Se utiliza StratifiedGroupKFold para separar lotes completos.
N_SPLITS_HOLDOUT = 5

# Fold utilizado como holdout.
FOLD_HOLDOUT = 1


# ============================================================
# RUTAS
# ============================================================

MODEL_DIR = "models"

CANDIDATE_DIR = (
    "models/candidate"
)

# ------------------------------------------------------------
# Modelo actualmente activo
# ------------------------------------------------------------

MODEL_PATH_PRODUCCION = (
    "models/model.pkl"
)

ENCODERS_PATH_PRODUCCION = (
    "models/encoders.pkl"
)

THRESHOLD_PATH_PRODUCCION = (
    "models/threshold_rf_sin_lote29.pkl"
)

# ------------------------------------------------------------
# Modelo candidato
# ------------------------------------------------------------

MODEL_PATH_CANDIDATO = (
    "models/candidate/model.pkl"
)

ENCODERS_PATH_CANDIDATO = (
    "models/candidate/encoders.pkl"
)

THRESHOLD_PATH_CANDIDATO = (
    "models/candidate/threshold_rf_sin_lote29.pkl"
)

# ------------------------------------------------------------
# Diagnósticos
# ------------------------------------------------------------

DIAGNOSTICO_LOTES_PATH = (
    "models/diagnostico_lotes_rf_sin_lote29.csv"
)

PREDICCIONES_TEST_PATH = (
    "models/predicciones_test_rf_sin_lote29.csv"
)

# ------------------------------------------------------------
# Diagnóstico de comparación
# ------------------------------------------------------------

COMPARACION_MODELOS_PATH = (
    "models/comparacion_modelos.csv"
)


# ============================================================
# BUSCAR MEJOR UMBRAL
# ============================================================

def mejor_umbral(
    y_true,
    y_proba,
    recall_minimo=RECALL_MINIMO
):

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
    # Caso normal
    # --------------------------------------------------------

    if candidatos:

        return max(
            candidatos,
            key=lambda x: x[1]
        )

    # --------------------------------------------------------
    # Caso excepcional
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
# ENCODERS
# ============================================================

def ajustar_encoders(X_train):

    X_train = X_train.copy()

    encoders = {}

    categorical_columns = (
        X_train
        .select_dtypes(
            include=[
                "object",
                "category"
            ]
        )
        .columns
    )

    for col in categorical_columns:

        encoder = LabelEncoder()

        valores = (
            X_train[col]
            .astype(str)
        )

        encoder.fit(valores)

        X_train[col] = (
            encoder.transform(
                valores
            )
        )

        encoders[col] = encoder

    return (
        X_train,
        encoders
    )


def transformar_features(
    X,
    encoders
):

    X = X.copy()

    for col, encoder in encoders.items():

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

    if "actividad_sensor_indice" in X.columns:

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
# RANDOM FOREST
# ============================================================

def crear_modelo():

    return RandomForestClassifier(

        n_estimators=800,

        max_depth=8,

        min_samples_split=8,

        min_samples_leaf=8,

        class_weight="balanced",

        random_state=RANDOM_STATE,

        n_jobs=-1
    )


# ============================================================
# UMBRAL OOF
# ============================================================

def obtener_umbral_oof(
    X_train,
    y_train,
    groups_train,
    random_state_repeat
):

    inner_kfold = StratifiedGroupKFold(

        n_splits=N_SPLITS_INTERNOS,

        shuffle=True,

        random_state=random_state_repeat
    )

    y_oof = np.zeros(
        len(y_train),
        dtype=float
    )

    for (
        inner_train_idx,
        inner_valid_idx
    ) in inner_kfold.split(
        X_train,
        y_train,
        groups_train
    ):

        X_inner_train = (
            X_train
            .iloc[inner_train_idx]
            .copy()
        )

        X_inner_valid = (
            X_train
            .iloc[inner_valid_idx]
            .copy()
        )

        y_inner_train = (
            y_train
            .iloc[inner_train_idx]
        )

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
        # Encoders
        # ----------------------------------------------------

        (
            X_inner_train,
            inner_encoders
        ) = ajustar_encoders(
            X_inner_train
        )

        X_inner_valid = (
            transformar_features(
                X_inner_valid,
                inner_encoders
            )
        )

        # ----------------------------------------------------
        # Modelo
        # ----------------------------------------------------

        model = crear_modelo()

        model.fit(
            X_inner_train,
            y_inner_train
        )

        # ----------------------------------------------------
        # OOF
        # ----------------------------------------------------

        proba = (
            model
            .predict_proba(
                X_inner_valid
            )[:, 1]
        )

        y_oof[
            inner_valid_idx
        ] = proba

    (
        umbral,
        precision,
        recall
    ) = mejor_umbral(
        y_train,
        y_oof,
        RECALL_MINIMO
    )

    return (
        umbral,
        precision,
        recall
    )


# ============================================================
# EVALUAR TEST
# ============================================================

def evaluar_test(
    y_test,
    y_proba,
    umbral
):

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

    matriz = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = (
        matriz.ravel()
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
# UNA REPETICIÓN
# ============================================================

def correr_una_repeticion(
    X,
    y,
    groups,
    metadata,
    repeat_idx,
    random_state_repeat
):

    group_kfold = StratifiedGroupKFold(

        n_splits=N_SPLITS_EXTERNOS,

        shuffle=True,

        random_state=random_state_repeat
    )

    resultados_repeticion = []

    predicciones = []

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

        # ----------------------------------------------------
        # TRAIN / TEST
        # ----------------------------------------------------

        X_train = (
            X
            .iloc[train_idx]
            .copy()
        )

        X_test = (
            X
            .iloc[test_idx]
            .copy()
        )

        y_train = (
            y
            .iloc[train_idx]
            .copy()
        )

        y_test = (
            y
            .iloc[test_idx]
            .copy()
        )

        groups_train = (
            groups
            .iloc[train_idx]
            .copy()
        )

        # ----------------------------------------------------
        # Umbral OOF
        # ----------------------------------------------------

        (
            umbral,
            precision_oof,
            recall_oof
        ) = obtener_umbral_oof(

            X_train,

            y_train,

            groups_train,

            random_state_repeat
        )

        # ----------------------------------------------------
        # Nulos
        # ----------------------------------------------------

        X_train = preparar_nulos(
            X_train
        )

        X_test = preparar_nulos(
            X_test
        )

        # ----------------------------------------------------
        # Encoders
        # ----------------------------------------------------

        (
            X_train_encoded,
            encoders_fold
        ) = ajustar_encoders(
            X_train
        )

        X_test_encoded = (
            transformar_features(
                X_test,
                encoders_fold
            )
        )

        # ----------------------------------------------------
        # Modelo
        # ----------------------------------------------------

        model = crear_modelo()

        model.fit(
            X_train_encoded,
            y_train
        )

        # ----------------------------------------------------
        # Probabilidades
        # ----------------------------------------------------

        y_proba = (
            model
            .predict_proba(
                X_test_encoded
            )[:, 1]
        )

        # ----------------------------------------------------
        # Evaluación
        # ----------------------------------------------------

        resultado = evaluar_test(

            y_test,

            y_proba,

            umbral
        )

        resultado["repeticion"] = (
            repeat_idx
        )

        resultado["fold"] = (
            fold
        )

        resultados_repeticion.append(
            resultado
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        metadata_test = (
            metadata
            .iloc[test_idx]
            .copy()
            .reset_index(drop=True)
        )

        y_test_array = (
            y_test.to_numpy()
        )

        y_pred_array = (
            y_proba >= umbral
        ).astype(int)

        # ----------------------------------------------------
        # Guardar predicciones
        # ----------------------------------------------------

        for i in range(
            len(metadata_test)
        ):

            predicciones.append({

                "repeticion":
                    repeat_idx,

                "fold":
                    fold,

                "id_lote":
                    metadata_test
                    .iloc[i]["id_lote"],

                "fecha":
                    metadata_test
                    .iloc[i]["fecha"],

                "y_true":
                    int(
                        y_test_array[i]
                    ),

                "y_proba":
                    float(
                        y_proba[i]
                    ),

                "y_pred":
                    int(
                        y_pred_array[i]
                    ),

                "umbral":
                    float(
                        umbral
                    )
            })

        # ----------------------------------------------------
        # Print reducido
        # ----------------------------------------------------

        print(

            f"R{repeat_idx} F{fold} | "

            f"AUC={resultado['auc']:.3f} | "

            f"P={resultado['precision']:.3f} | "

            f"R={resultado['recall']:.3f} | "

            f"F1={resultado['f1']:.3f}"
        )

    return (
        resultados_repeticion,
        predicciones
    )


# ============================================================
# AGREGAR PREDICCIONES
# ============================================================

def agregar_predicciones(
    predicciones
):

    df_pred = pd.DataFrame(
        predicciones
    )

    if df_pred.empty:

        return df_pred

    return (
        df_pred
        .groupby(
            [
                "id_lote",
                "fecha"
            ],
            as_index=False
        )
        .agg(

            y_true=(
                "y_true",
                "first"
            ),

            y_proba=(
                "y_proba",
                "mean"
            ),

            n_repeticiones=(
                "repeticion",
                "count"
            )
        )
    )


# ============================================================
# AUC POR LOTE
# ============================================================

def analizar_lotes_agregado(
    df_agregado
):

    if df_agregado.empty:

        return pd.DataFrame()

    resultados = []

    for (
        id_lote,
        grupo
    ) in df_agregado.groupby(
        "id_lote"
    ):

        y_true = (
            grupo["y_true"]
            .to_numpy()
        )

        y_proba = (
            grupo["y_proba"]
            .to_numpy()
        )

        n = len(grupo)

        positivos = int(
            np.sum(
                y_true == 1
            )
        )

        negativos = int(
            np.sum(
                y_true == 0
            )
        )

        if (
            positivos > 0
            and negativos > 0
        ):

            auc = roc_auc_score(
                y_true,
                y_proba
            )

        else:

            auc = np.nan

        resultados.append({

            "id_lote":
                id_lote,

            "n_observaciones":
                n,

            "positivos":
                positivos,

            "negativos":
                negativos,

            "porcentaje_positivo":
                positivos / n,

            "auc":
                auc
        })

    return (
        pd.DataFrame(
            resultados
        )
        .sort_values(
            "auc",
            ascending=True,
            na_position="last"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# MOSTRAR AUC POR LOTE
# ============================================================

def mostrar_analisis_lotes(
    df_lotes
):

    if df_lotes.empty:

        return

    print(
        "\n" + "=" * 70
    )

    print(
        "📊 AUC POR LOTE"
    )

    print(
        "=" * 70
    )

    print(
        f"{'Lote':<12}"
        f"{'N':>8}"
        f"{'Pos':>8}"
        f"{'Neg':>8}"
        f"{'%Pos':>9}"
        f"{'AUC':>10}"
    )

    print(
        "-" * 65
    )

    for _, row in (
        df_lotes.iterrows()
    ):

        auc_text = (

            f"{row['auc']:.3f}"

            if pd.notna(
                row["auc"]
            )

            else "N/A"
        )

        print(

            f"{row['id_lote']:<12}"

            f"{int(row['n_observaciones']):>8}"

            f"{int(row['positivos']):>8}"

            f"{int(row['negativos']):>8}"

            f"{row['porcentaje_positivo'] * 100:>8.1f}%"

            f"{auc_text:>10}"
        )


# ============================================================
# CARGAR MODELO DE PRODUCCIÓN
# ============================================================

def cargar_modelo_produccion():

    if not os.path.exists(
        MODEL_PATH_PRODUCCION
    ):

        raise FileNotFoundError(

            "❌ No existe el modelo "
            f"de producción: "
            f"{MODEL_PATH_PRODUCCION}"
        )

    if not os.path.exists(
        ENCODERS_PATH_PRODUCCION
    ):

        raise FileNotFoundError(

            "❌ No existen los encoders "
            f"de producción: "
            f"{ENCODERS_PATH_PRODUCCION}"
        )

    if not os.path.exists(
        THRESHOLD_PATH_PRODUCCION
    ):

        raise FileNotFoundError(

            "❌ No existe el threshold "
            f"de producción: "
            f"{THRESHOLD_PATH_PRODUCCION}"
        )

    with open(
        MODEL_PATH_PRODUCCION,
        "rb"
    ) as f:

        modelo = pickle.load(f)

    with open(
        ENCODERS_PATH_PRODUCCION,
        "rb"
    ) as f:

        encoders = pickle.load(f)

    with open(
        THRESHOLD_PATH_PRODUCCION,
        "rb"
    ) as f:

        threshold = pickle.load(f)

    return (
        modelo,
        encoders,
        threshold
    )


# ============================================================
# EVALUAR MODELO SOBRE UN CONJUNTO
# ============================================================

def evaluar_modelo_sobre_datos(

    modelo,

    encoders,

    threshold,

    X_test,

    y_test
):

    X_test = preparar_nulos(
        X_test
    )

    X_test_encoded = (
        transformar_features(
            X_test,
            encoders
        )
    )

    y_proba = (
        modelo
        .predict_proba(
            X_test_encoded
        )[:, 1]
    )

    resultado = evaluar_test(

        y_test,

        y_proba,

        threshold
    )

    return (
        resultado,
        y_proba
    )


# ============================================================
# CREAR HOLDOUT DE PRODUCCIÓN
# ============================================================

def separar_holdout_produccion(
    df_produccion
):

    if df_produccion.empty:

        raise ValueError(

            "❌ No existen datos de "
            "producción verificados."
        )

    groups = (
        df_produccion[
            "id_lote"
        ]
    )

    y = (
        df_produccion[
            TARGET
        ]
        .astype(int)
    )

    if groups.nunique() < N_SPLITS_HOLDOUT:

        raise ValueError(

            "❌ No hay suficientes lotes "
            "de producción para crear "
            "el holdout."
        )

    kfold = StratifiedGroupKFold(

        n_splits=N_SPLITS_HOLDOUT,

        shuffle=True,

        random_state=RANDOM_STATE
    )

    splits = list(
        kfold.split(
            df_produccion,
            y,
            groups
        )
    )

    if (
        FOLD_HOLDOUT < 1
        or FOLD_HOLDOUT > len(splits)
    ):

        raise ValueError(
            "❌ FOLD_HOLDOUT inválido."
        )

    _, holdout_idx = (
        splits[FOLD_HOLDOUT - 1]
    )

    mask_holdout = (
        df_produccion
        .index
        .isin(holdout_idx)
    )

    df_holdout = (
        df_produccion[
            mask_holdout
        ]
        .copy()
        .reset_index(drop=True)
    )

    df_train_produccion = (
        df_produccion[
            ~mask_holdout
        ]
        .copy()
        .reset_index(drop=True)
    )

    return (
        df_train_produccion,
        df_holdout
    )


# ============================================================
# COMPARAR MODELO ACTUAL VS CANDIDATO
# ============================================================

def candidato_supera_actual(

    resultado_actual,

    resultado_candidato
):

    # --------------------------------------------------------
    # Regla 1:
    # El candidato debe cumplir el Recall mínimo.
    # --------------------------------------------------------

    if (
        resultado_candidato["recall"]
        < RECALL_MINIMO
    ):

        return False

    # --------------------------------------------------------
    # Regla 2:
    # No permitir una caída excesiva de Recall.
    # --------------------------------------------------------

    if (
        resultado_candidato["recall"]
        <
        (
            resultado_actual["recall"]
            - TOLERANCIA_RECALL
        )
    ):

        return False

    # --------------------------------------------------------
    # Regla 3:
    # No permitir una caída excesiva de AUC.
    # --------------------------------------------------------

    if (
        resultado_candidato["auc"]
        <
        (
            resultado_actual["auc"]
            - TOLERANCIA_AUC
        )
    ):

        return False

    # --------------------------------------------------------
    # Regla 4:
    # El F1 debe mejorar.
    # --------------------------------------------------------

    if (
        resultado_candidato["f1"]
        <=
        resultado_actual["f1"]
    ):

        return False

    return True


# ============================================================
# GUARDAR CANDIDATO
# ============================================================

def guardar_candidato(

    modelo,

    encoders,

    threshold
):

    os.makedirs(
        CANDIDATE_DIR,
        exist_ok=True
    )

    with open(
        MODEL_PATH_CANDIDATO,
        "wb"
    ) as f:

        pickle.dump(
            modelo,
            f
        )

    with open(
        ENCODERS_PATH_CANDIDATO,
        "wb"
    ) as f:

        pickle.dump(
            encoders,
            f
        )

    with open(
        THRESHOLD_PATH_CANDIDATO,
        "wb"
    ) as f:

        pickle.dump(
            threshold,
            f
        )

    print(
        "\n✓ Candidato guardado:"
    )

    print(
        f"  Modelo: "
        f"{MODEL_PATH_CANDIDATO}"
    )

    print(
        f"  Encoders: "
        f"{ENCODERS_PATH_CANDIDATO}"
    )

    print(
        f"  Threshold: "
        f"{THRESHOLD_PATH_CANDIDATO}"
    )


# ============================================================
# PROMOVER CANDIDATO
# ============================================================

def promover_candidato():

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    shutil.copy2(

        MODEL_PATH_CANDIDATO,

        MODEL_PATH_PRODUCCION
    )

    shutil.copy2(

        ENCODERS_PATH_CANDIDATO,

        ENCODERS_PATH_PRODUCCION
    )

    shutil.copy2(

        THRESHOLD_PATH_CANDIDATO,

        THRESHOLD_PATH_PRODUCCION
    )

    print(
        "\n🚀 CANDIDATO PROMOVIDO"
    )

    print(
        f"✓ Modelo activo: "
        f"{MODEL_PATH_PRODUCCION}"
    )

    print(
        f"✓ Encoders activos: "
        f"{ENCODERS_PATH_PRODUCCION}"
    )

    print(
        f"✓ Threshold activo: "
        f"{THRESHOLD_PATH_PRODUCCION}"
    )


# ============================================================
# GUARDAR COMPARACIÓN
# ============================================================

def guardar_comparacion(

    resultado_actual,

    resultado_candidato,

    promovido
):

    df_comparacion = pd.DataFrame([

        {
            "modelo":
                "produccion",

            **resultado_actual
        },

        {
            "modelo":
                "candidato",

            **resultado_candidato
        }
    ])

    df_comparacion[
        "promovido"
    ] = False

    if promovido:

        df_comparacion.loc[
            df_comparacion["modelo"]
            == "candidato",
            "promovido"
        ] = True

    df_comparacion.to_csv(

        COMPARACION_MODELOS_PATH,

        index=False,

        encoding="utf-8"
    )


# ============================================================
# ENTRENAMIENTO PRINCIPAL
# ============================================================

def entrenar_modelo():

    print(
        "🐄 SIPREM-BOVINO — "
        "RANDOM FOREST"
    )

    print(
        f"🎯 Target: {TARGET}"
    )

    print(
        f"🚫 Lote excluido: "
        f"{LOTE_EXCLUIDO}"
    )

    # ========================================================
    # CONEXIÓN
    # ========================================================

    from src.config.database import (
        get_connection
    )

    conn = get_connection()

    try:

        # ----------------------------------------------------
        # Cargar histórico + producción
        # ----------------------------------------------------

        query = """

            SELECT

                id_lote,
                fecha,
                distrito,
                categoria_zootecnica,
                raza_predominante,
                lote_sensorizado,
                distancia_centro_veterinario_km,
                altitud_msnm,
                tamano_lote_cabezas,
                uso_registro_digital,
                cobertura_vacunacion_pct,
                dias_desde_desparasitacion,
                animales_nuevos_30d,
                casos_respiratorios,
                casos_diarreicos,
                temperatura_min_c,
                temperatura_media_c,
                temperatura_max_c,
                humedad_relativa_pct,
                precipitacion_semanal_mm,
                condicion_pastura_indice,
                indice_ndvi_satelital,
                consumo_ms_kg_animal_dia,
                agua_l_animal_dia,
                actividad_sensor_indice,
                condicion_corporal_prom,
                precio_leche_local_s_kg,
                semana_sin,
                semana_cos,
                media_movil_4s_pastura,
                media_movil_4s_condicion_corporal,
                media_movil_4s_temperatura,
                target_riesgo_alto_4sem,

                'dataset_features'
                AS fuente_dato

            FROM gold_ml.dataset_features

            UNION ALL

            SELECT

                id_lote,
                fecha,
                distrito,
                categoria_zootecnica,
                raza_predominante,
                lote_sensorizado,
                distancia_centro_veterinario_km,
                altitud_msnm,
                tamano_lote_cabezas,
                uso_registro_digital,
                cobertura_vacunacion_pct,
                dias_desde_desparasitacion,
                animales_nuevos_30d,
                casos_respiratorios,
                casos_diarreicos,
                temperatura_min_c,
                temperatura_media_c,
                temperatura_max_c,
                humedad_relativa_pct,
                precipitacion_semanal_mm,
                condicion_pastura_indice,
                indice_ndvi_satelital,
                consumo_ms_kg_animal_dia,
                agua_l_animal_dia,
                actividad_sensor_indice,
                condicion_corporal_prom,
                precio_leche_local_s_kg,
                semana_sin,
                semana_cos,
                media_movil_4s_pastura,
                media_movil_4s_condicion_corporal,
                media_movil_4s_temperatura,
                target_riesgo_alto_4sem,

                'produccion_verificada_sin_intervencion'
                AS fuente_dato

            FROM
                 gold_ml.predicciones_verificadas_sin_intervencion

            ORDER BY
                id_lote,
                fecha
        """

        df = pd.read_sql(
            query,
            conn
        )

    finally:

        conn.close()

    # ========================================================
    # DIAGNÓSTICO DE FUENTES
    # ========================================================

    print(
        "\n✓ Filas por fuente:"
    )

    print(
        df[
            "fuente_dato"
        ]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # DIAGNÓSTICO ORIGINAL
    # ========================================================

    filas_originales = len(df)

    lotes_originales = (
        df[
            "id_lote"
        ]
        .nunique()
    )

    nulos_originales = (
        df[TARGET]
        .isna()
        .sum()
    )

    positivos_originales = (
        df[TARGET]
        .eq(True)
        .sum()
    )

    negativos_originales = (
        df[TARGET]
        .eq(False)
        .sum()
    )

    # ========================================================
    # EXCLUIR LOTE
    # ========================================================

    if LOTE_EXCLUIDO not in (
        df[
            "id_lote"
        ].unique()
    ):

        raise ValueError(

            f"❌ El lote "
            f"'{LOTE_EXCLUIDO}' "
            f"no existe en el dataset."
        )

    filas_lote_excluido = (
        df[
            "id_lote"
        ]
        .eq(LOTE_EXCLUIDO)
        .sum()
    )

    df = (
        df[
            df["id_lote"]
            != LOTE_EXCLUIDO
        ]
        .copy()
        .reset_index(drop=True)
    )

    # ========================================================
    # SEPARAR PRODUCCIÓN
    # ========================================================

    df_produccion = (
        df[
            (
                df[
                    "fuente_dato"
                ]
                ==
                "produccion_verificada_sin_intervencion"
            )
            &
            (
                df[TARGET]
                .notna()
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    df_historico = (
        df[
            df[
                "fuente_dato"
            ]
            ==
            "dataset_features"
        ]
        .copy()
        .reset_index(drop=True)
    )

    # ========================================================
    # VERIFICAR DUPLICADOS
    # ========================================================

    duplicados = (
        df
        .duplicated(
            subset=[
                "id_lote",
                "fecha"
            ],
            keep=False
        )
        .sum()
    )

    print(
        f"\n✓ Filas potencialmente "
        f"duplicadas por lote/fecha: "
        f"{duplicados}"
    )

    # ========================================================
    # CREAR HOLDOUT DE PRODUCCIÓN
    # ========================================================

    (
        df_produccion_train,
        df_produccion_holdout
    ) = separar_holdout_produccion(
        df_produccion
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "📦 SEPARACIÓN DE PRODUCCIÓN"
    )

    print(
        "=" * 70
    )

    print(
        f"✓ Producción total: "
        f"{len(df_produccion)} filas"
    )

    print(
        f"✓ Producción entrenamiento: "
        f"{len(df_produccion_train)} filas"
    )

    print(
        f"✓ Producción holdout: "
        f"{len(df_produccion_holdout)} filas"
    )

    print(
        f"✓ Lotes producción entrenamiento: "
        f"{df_produccion_train['id_lote'].nunique()}"
    )

    print(
        f"✓ Lotes producción holdout: "
        f"{df_produccion_holdout['id_lote'].nunique()}"
    )

    # ========================================================
    # DATASET PARA ENTRENAR CANDIDATO
    # ========================================================

    df_entrenamiento = pd.concat(
        [
            df_historico,
            df_produccion_train
        ],
        ignore_index=True
    )

    # Solo filas con target conocido.
    df_entrenamiento = (
        df_entrenamiento[
            df_entrenamiento[
                TARGET
            ].notna()
        ]
        .copy()
        .reset_index(drop=True)
    )

    # ========================================================
    # METADATA
    # ========================================================

    metadata = (
        df_entrenamiento[
            [
                "id_lote",
                "fecha"
            ]
        ]
        .copy()
        .reset_index(drop=True)
    )

    # ========================================================
    # GROUPS
    # ========================================================

    groups = (
        df_entrenamiento[
            "id_lote"
        ]
        .copy()
        .reset_index(drop=True)
    )

    # ========================================================
    # X / Y
    # ========================================================

    X = (
        df_entrenamiento[
            FEATURES
        ]
        .copy()
        .reset_index(drop=True)
    )

    y = (
        df_entrenamiento[
            TARGET
        ]
        .astype(int)
        .reset_index(drop=True)
    )

    # ========================================================
    # RESUMEN
    # ========================================================

    print(
        "\n" +
        f"✓ Dataset original: "
        f"{filas_originales} filas / "
        f"{lotes_originales} lotes"
    )

    print(
        f"✓ Filas de {LOTE_EXCLUIDO}: "
        f"{filas_lote_excluido}"
    )

    print(
        f"✓ Histórico utilizado: "
        f"{len(df_historico)} filas"
    )

    print(
        f"✓ Producción utilizada "
        f"para entrenamiento: "
        f"{len(df_produccion_train)} filas"
    )

    print(
        f"✓ Holdout producción: "
        f"{len(df_produccion_holdout)} filas"
    )

    print(
        f"✓ Dataset candidato: "
        f"{len(df_entrenamiento)} filas / "
        f"{df_entrenamiento['id_lote'].nunique()} lotes"
    )

    print(
        f"✓ Target nulo original: "
        f"{nulos_originales}"
    )

    print(
        f"✓ Positivos originales: "
        f"{positivos_originales}"
    )

    print(
        f"✓ Negativos originales: "
        f"{negativos_originales}"
    )

    # ========================================================
    # CONFIGURACIÓN
    # ========================================================

    print(
        "\n" +
        f"✓ Validación interna: "
        f"{N_REPEATS} × "
        f"{N_SPLITS_EXTERNOS}"
    )

    print(
        f"✓ Lotes utilizados: "
        f"{groups.nunique()}"
    )

    print(
        f"✓ Features: "
        f"{len(FEATURES)}"
    )

    print(
        f"✓ Recall mínimo: "
        f"{RECALL_MINIMO:.2f}"
    )

    # ========================================================
    # EVALUACIÓN DEL CANDIDATO
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "📊 VALIDACIÓN DEL CANDIDATO"
    )

    print(
        "=" * 70
    )

    todos_los_resultados = []

    todas_las_predicciones = []

    for repeat_idx in range(
        1,
        N_REPEATS + 1
    ):

        random_state_repeat = (
            RANDOM_STATE
            + repeat_idx
        )

        (
            resultados_repeticion,
            predicciones
        ) = correr_una_repeticion(

            X,

            y,

            groups,

            metadata,

            repeat_idx,

            random_state_repeat
        )

        todos_los_resultados.extend(
            resultados_repeticion
        )

        todas_las_predicciones.extend(
            predicciones
        )

    # ========================================================
    # RESULTADOS GLOBALES
    # ========================================================

    df_resultados = pd.DataFrame(
        todos_los_resultados
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "📊 RESULTADOS CANDIDATO"
    )

    print(
        "=" * 70
    )

    print(
        f"AUC:       "
        f"{df_resultados['auc'].mean():.3f} ± "
        f"{df_resultados['auc'].std(ddof=0):.3f}"
    )

    print(
        f"Precision:  "
        f"{df_resultados['precision'].mean():.3f} ± "
        f"{df_resultados['precision'].std(ddof=0):.3f}"
    )

    print(
        f"Recall:     "
        f"{df_resultados['recall'].mean():.3f} ± "
        f"{df_resultados['recall'].std(ddof=0):.3f}"
    )

    print(
        f"F1:         "
        f"{df_resultados['f1'].mean():.3f} ± "
        f"{df_resultados['f1'].std(ddof=0):.3f}"
    )

    # ========================================================
    # PROMEDIO POR REPETICIÓN
    # ========================================================

    print(
        "\n📋 POR REPETICIÓN"
    )

    print(
        f"{'Rep':<8}"
        f"{'AUC':>10}"
        f"{'Prec':>10}"
        f"{'Recall':>10}"
        f"{'F1':>10}"
    )

    print(
        "-" * 48
    )

    for repeat_idx, grupo in (
        df_resultados
        .groupby("repeticion")
    ):

        print(

            f"{repeat_idx:<8}"

            f"{grupo['auc'].mean():>10.3f}"

            f"{grupo['precision'].mean():>10.3f}"

            f"{grupo['recall'].mean():>10.3f}"

            f"{grupo['f1'].mean():>10.3f}"
        )

    # ========================================================
    # PREDICCIONES AGREGADAS
    # ========================================================

    df_predicciones_agregadas = (
        agregar_predicciones(
            todas_las_predicciones
        )
    )

    # ========================================================
    # AUC POR LOTE
    # ========================================================

    df_lotes = (
        analizar_lotes_agregado(
            df_predicciones_agregadas
        )
    )

    mostrar_analisis_lotes(
        df_lotes
    )

    # ========================================================
    # GUARDAR DIAGNÓSTICOS
    # ========================================================

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    df_lotes.to_csv(

        DIAGNOSTICO_LOTES_PATH,

        index=False,

        encoding="utf-8"
    )

    df_predicciones_agregadas.to_csv(

        PREDICCIONES_TEST_PATH,

        index=False,

        encoding="utf-8"
    )

    # ========================================================
    # ENTRENAMIENTO DEL CANDIDATO
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "🚀 ENTRENANDO CANDIDATO"
    )

    print(
        "=" * 70
    )

    X_final = preparar_nulos(
        X
    )

    (
        X_final_encoded,
        encoders_final
    ) = ajustar_encoders(
        X_final
    )

    modelo_candidato = (
        crear_modelo()
    )

    modelo_candidato.fit(

        X_final_encoded,

        y
    )

    # ========================================================
    # UMBRAL FINAL DEL CANDIDATO
    # ========================================================

    (
        umbral_candidato,
        precision_oof_candidato,
        recall_oof_candidato
    ) = obtener_umbral_oof(

        X,

        y,

        groups,

        RANDOM_STATE
    )

    print(
        f"\n✓ Threshold candidato: "
        f"{umbral_candidato:.2f}"
    )

    print(
        f"✓ OOF Precision candidato: "
        f"{precision_oof_candidato:.3f}"
    )

    print(
        f"✓ OOF Recall candidato: "
        f"{recall_oof_candidato:.3f}"
    )

    # ========================================================
    # GUARDAR CANDIDATO
    # ========================================================

    guardar_candidato(

        modelo_candidato,

        encoders_final,

        umbral_candidato
    )

    # ========================================================
    # CARGAR MODELO ACTUAL
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "🔎 COMPARACIÓN "
        "PRODUCCIÓN VS CANDIDATO"
    )

    print(
        "=" * 70
    )

    (
        modelo_actual,
        encoders_actual,
        threshold_actual
    ) = cargar_modelo_produccion()

    # ========================================================
    # CREAR HOLDOUT X / Y
    # ========================================================

    X_holdout = (
        df_produccion_holdout[
            FEATURES
        ]
        .copy()
        .reset_index(drop=True)
    )

    y_holdout = (
        df_produccion_holdout[
            TARGET
        ]
        .astype(int)
        .reset_index(drop=True)
    )

    # ========================================================
    # EVALUAR MODELO ACTUAL
    # ========================================================

    (
        resultado_actual,
        _
    ) = evaluar_modelo_sobre_datos(

        modelo_actual,

        encoders_actual,

        threshold_actual,

        X_holdout,

        y_holdout
    )

    # ========================================================
    # EVALUAR CANDIDATO
    # ========================================================

    (
        resultado_candidato,
        _
    ) = evaluar_modelo_sobre_datos(

        modelo_candidato,

        encoders_final,

        umbral_candidato,

        X_holdout,

        y_holdout
    )

    # ========================================================
    # MOSTRAR COMPARACIÓN
    # ========================================================

    print(
        "\n📊 MODELO ACTUAL"
    )

    print(
        f"AUC:       "
        f"{resultado_actual['auc']:.3f}"
    )

    print(
        f"Precision: "
        f"{resultado_actual['precision']:.3f}"
    )

    print(
        f"Recall:    "
        f"{resultado_actual['recall']:.3f}"
    )

    print(
        f"F1:        "
        f"{resultado_actual['f1']:.3f}"
    )

    print(
        f"Threshold: "
        f"{resultado_actual['umbral']:.2f}"
    )

    print(
        "\n🆕 CANDIDATO"
    )

    print(
        f"AUC:       "
        f"{resultado_candidato['auc']:.3f}"
    )

    print(
        f"Precision: "
        f"{resultado_candidato['precision']:.3f}"
    )

    print(
        f"Recall:    "
        f"{resultado_candidato['recall']:.3f}"
    )

    print(
        f"F1:        "
        f"{resultado_candidato['f1']:.3f}"
    )

    print(
        f"Threshold: "
        f"{resultado_candidato['umbral']:.2f}"
    )

    # ========================================================
    # DECISIÓN
    # ========================================================

    promovido = candidato_supera_actual(

        resultado_actual,

        resultado_candidato
    )

    # ========================================================
    # GUARDAR COMPARACIÓN
    # ========================================================

    guardar_comparacion(

        resultado_actual,

        resultado_candidato,

        promovido
    )

    # ========================================================
    # PROMOCIÓN / RECHAZO
    # ========================================================

    if promovido:

        print(
            "\n" + "=" * 70
        )

        print(
            "✅ EL CANDIDATO CUMPLE "
            "LOS CRITERIOS"
        )

        print(
            "=" * 70
        )

        promover_candidato()

    else:

        print(
            "\n" + "=" * 70
        )

        print(
            "⚠️ CANDIDATO NO PROMOVIDO"
        )

        print(
            "=" * 70
        )

        print(
            "✓ El modelo de producción "
            "se mantiene sin cambios."
        )

        print(
            "✓ El candidato permanece "
            "en models/candidate/."
        )

    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "🏁 PROCESO FINALIZADO"
    )

    print(
        "=" * 70
    )

    print(
        f"✓ Diagnóstico: "
        f"{DIAGNOSTICO_LOTES_PATH}"
    )

    print(
        f"✓ Predicciones CV: "
        f"{PREDICCIONES_TEST_PATH}"
    )

    print(
        f"✓ Comparación: "
        f"{COMPARACION_MODELOS_PATH}"
    )

    print(
        f"✓ Candidato: "
        f"{CANDIDATE_DIR}"
    )

    if promovido:

        print(
            "✓ Estado: "
            "CANDIDATO PROMOVIDO"
        )

    else:

        print(
            "✓ Estado: "
            "MODELO ACTUAL CONSERVADO"
        )

    return modelo_candidato


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    entrenar_modelo()