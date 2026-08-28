# ============================================================
# SIPREM-BOVINO
# RANDOM FOREST — TARGET DE 4 SEMANAS
#
# EXPERIMENTO:
#
# Se excluye explícitamente:
#
#     LOTE_29
#
# El objetivo es medir cómo cambia el rendimiento del modelo
# cuando este lote no participa en entrenamiento ni evaluación.
#
# ============================================================


# ============================================================
# IMPORTACIONES
# ============================================================

import os
import pickle

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
# LOTE EXCLUIDO
# ------------------------------------------------------------

LOTE_EXCLUIDO = "LOTE_29"


# ============================================================
# RUTAS
# ============================================================

DIAGNOSTICO_LOTES_PATH = (
    "models/diagnostico_lotes_rf_sin_lote29.csv"
)

PREDICCIONES_TEST_PATH = (
    "models/predicciones_test_rf_sin_lote29.csv"
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
            include=["object", "category"]
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

    return X_train, encoders


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

        X["actividad_sensor_indice"] = (
            X["actividad_sensor_indice"]
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
        # PRINT REDUCIDO
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
# ENTRENAMIENTO PRINCIPAL
# ============================================================

def entrenar_modelo():

    print(
        "🐄 SIPREM-BOVINO — RANDOM FOREST"
    )

    print(
        f"🎯 Target: {TARGET}"
    )

    print(
        f"🚫 Lote excluido: {LOTE_EXCLUIDO}"
    )

    # ========================================================
    # CONEXIÓN
    # ========================================================

    from src.config.database import get_connection

    conn = get_connection()

    try:

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

    # ========================================================
    # DIAGNÓSTICO ORIGINAL
    # ========================================================

    filas_originales = len(df)

    lotes_originales = (
        df["id_lote"].nunique()
    )

    nulos_originales = (
        df[TARGET].isna().sum()
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
        df["id_lote"].unique()
    ):

        raise ValueError(

            f"❌ El lote "
            f"'{LOTE_EXCLUIDO}' "
            f"no existe en el dataset."
        )

    filas_lote_excluido = (
        df["id_lote"]
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
    # FILAS ETIQUETADAS
    # ========================================================

    df_entrenamiento = (

        df[
            df[TARGET].notna()
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
    # RESUMEN DEL DATASET
    # ========================================================

    print(
        "\n"
        f"✓ Dataset original: "
        f"{filas_originales} filas / "
        f"{lotes_originales} lotes"
    )

    print(
        f"✓ Filas de {LOTE_EXCLUIDO}: "
        f"{filas_lote_excluido}"
    )

    print(
        f"✓ Dataset utilizado: "
        f"{len(df)} filas / "
        f"{df['id_lote'].nunique()} lotes"
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

    print(
        f"✓ Filas etiquetadas utilizadas: "
        f"{len(df_entrenamiento)}"
    )

    # ========================================================
    # CONFIGURACIÓN
    # ========================================================

    print(
        "\n"
        f"✓ Validación: "
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
        f"✓ Recall mínimo OOF: "
        f"{RECALL_MINIMO:.2f}"
    )

    # ========================================================
    # EVALUACIÓN
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "📊 EVALUACIÓN — 34 LOTES"
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
        "📊 RESULTADOS GLOBALES"
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

        f"Precision: "
        f"{df_resultados['precision'].mean():.3f} ± "
        f"{df_resultados['precision'].std(ddof=0):.3f}"
    )

    print(

        f"Recall:    "
        f"{df_resultados['recall'].mean():.3f} ± "
        f"{df_resultados['recall'].std(ddof=0):.3f}"
    )

    print(

        f"F1:        "
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
        "models",
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
    # ENTRENAMIENTO FINAL
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        f"🚀 MODELO FINAL — SIN {LOTE_EXCLUIDO}"
    )

    print(
        "=" * 70
    )

    X_final = preparar_nulos(
        X
    )

    (
        X_final,
        encoders_final
    ) = ajustar_encoders(
        X_final
    )

    modelo_final = crear_modelo()

    modelo_final.fit(
        X_final,
        y
    )

    # ========================================================
    # UMBRAL FINAL
    # ========================================================

    (
        umbral_final,
        precision_oof_final,
        recall_oof_final
    ) = obtener_umbral_oof(

        X,

        y,

        groups,

        RANDOM_STATE
    )

    print(

        f"✓ Umbral final: "
        f"{umbral_final:.2f}"
    )

    print(

        f"✓ OOF Precision: "
        f"{precision_oof_final:.3f}"
    )

    print(

        f"✓ OOF Recall: "
        f"{recall_oof_final:.3f}"
    )

    # ========================================================
    # GUARDAR
    # ========================================================

    threshold_path = (
        "models/threshold_rf_sin_lote29.pkl"
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

    with open(
        threshold_path,
        "wb"
    ) as f:

        pickle.dump(
            umbral_final,
            f
        )

    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n✅ ENTRENAMIENTO FINALIZADO"
    )

    print(
        f"Modelo: {MODEL_PATH}"
    )

    print(
        f"Encoders: {ENCODERS_PATH}"
    )

    print(
        f"Threshold: {threshold_path}"
    )

    print(
        f"Diagnóstico: "
        f"{DIAGNOSTICO_LOTES_PATH}"
    )

    print(
        f"Predicciones: "
        f"{PREDICCIONES_TEST_PATH}"
    )

    return modelo_final


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    entrenar_modelo()