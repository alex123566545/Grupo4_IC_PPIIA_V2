# ============================================================
# SIPREM-BOVINO
# RANDOM FOREST — TARGET DE 4 SEMANAS
#
# VALIDACIÓN EXTERNA:
#
#     GroupShuffleSplit
#
# VALIDACIÓN INTERNA:
#
#     StratifiedGroupKFold
#
# El objetivo de esta versión es evaluar el modelo mediante
# separaciones aleatorias por LOTE, evitando que las filas
# pertenecientes al mismo lote aparezcan simultáneamente
# en entrenamiento y test.
#
# Target:
#
#     target_riesgo_alto_4sem
#
# El target representa si existe al menos un episodio de
# riesgo de mortalidad alto dentro de la ventana de 4 semanas
# definida en la capa GOLD.
#
# ============================================================


# ============================================================
# IMPORTACIÓN DE LIBRERÍAS
# ============================================================

import os
import pickle

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold
)

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

# ------------------------------------------------------------
# VALIDACIÓN EXTERNA
# ------------------------------------------------------------
#
# GroupShuffleSplit realizará varias separaciones aleatorias.
#
# TEST_SIZE = 0.20 significa aproximadamente:
#
#     80% de los LOTES -> TRAIN
#     20% de los LOTES -> TEST
#
# La separación se realiza por id_lote, NO por filas.
#
# ------------------------------------------------------------

N_SPLITS_EXTERNOS = 5

TEST_SIZE = 0.20


# ------------------------------------------------------------
# VALIDACIÓN INTERNA
# ------------------------------------------------------------
#
# Dentro de cada TRAIN externo se utiliza
# StratifiedGroupKFold para obtener predicciones OOF.
#
# Estas predicciones se utilizan únicamente para
# seleccionar el umbral.
#
# ------------------------------------------------------------

N_SPLITS_INTERNOS = 4


# ------------------------------------------------------------
# RECALL MÍNIMO
# ------------------------------------------------------------
#
# Se busca maximizar Precision manteniendo:
#
#     Recall >= 0.65
#
# ------------------------------------------------------------

RECALL_MINIMO = 0.65


# ============================================================
# BUSCAR MEJOR UMBRAL
# ============================================================
#
# Busca el umbral que proporcione la mayor Precision
# siempre que Recall sea como mínimo RECALL_MINIMO.
#
# El umbral se obtiene exclusivamente a partir de
# predicciones OOF del conjunto TRAIN.
#
# El TEST externo jamás participa en esta decisión.
#
# ============================================================

def mejor_umbral(
    y_true,
    y_proba,
    recall_minimo=0.65
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
    # Si existen umbrales que cumplen el Recall mínimo,
    # elegir el que tenga mayor Precision.
    # --------------------------------------------------------

    if candidatos:

        return max(
            candidatos,
            key=lambda x: x[1]
        )

    # --------------------------------------------------------
    # Si ningún umbral consigue el Recall mínimo,
    # seleccionar el que produzca mayor Recall.
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
            encoder.transform(valores)
        )

        encoders[col] = encoder

    return X_train, encoders


# ============================================================
# TRANSFORMAR FEATURES
# ============================================================

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
# CREAR MODELO
# ============================================================

def crear_modelo():

    return RandomForestClassifier(

        n_estimators=800,

        max_depth=8,

        min_samples_split=5,

        min_samples_leaf=5,

        class_weight="balanced",

        random_state=RANDOM_STATE,

        n_jobs=-1
    )


# ============================================================
# OBTENER UMBRAL MEDIANTE OOF
# ============================================================
#
# Esta función trabaja únicamente con el TRAIN externo.
#
# Se utiliza StratifiedGroupKFold para:
#
# 1. Mantener los lotes separados.
# 2. Intentar mantener la proporción de clases.
# 3. Generar predicciones OOF.
#
# Posteriormente se utiliza toda la predicción OOF para
# encontrar el mejor umbral.
#
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

    for inner_train_idx, inner_valid_idx in (
        inner_kfold.split(
            X_train,
            y_train,
            groups_train
        )
    ):

        # ----------------------------------------------------
        # Separar TRAIN y VALIDACIÓN interna
        # ----------------------------------------------------

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
            .copy()
        )

        # ----------------------------------------------------
        # Preparar nulos
        # ----------------------------------------------------

        X_inner_train = (
            preparar_nulos(
                X_inner_train
            )
        )

        X_inner_valid = (
            preparar_nulos(
                X_inner_valid
            )
        )

        # ----------------------------------------------------
        # Encoders
        #
        # IMPORTANTE:
        #
        # Los encoders se ajustan únicamente con el
        # entrenamiento interno.
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
        # Crear y entrenar modelo
        # ----------------------------------------------------

        model = crear_modelo()

        model.fit(
            X_inner_train,
            y_inner_train
        )

        # ----------------------------------------------------
        # Predicciones OOF
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

    # --------------------------------------------------------
    # Seleccionar umbral usando SOLO OOF
    # --------------------------------------------------------

    (
        umbral,
        precision,
        recall
    ) = mejor_umbral(
        y_train,
        y_oof,
        RECALL_MINIMO
    )

    print(
        f"      Umbral OOF: {umbral:.2f} | "
        f"Precision OOF: {precision:.3f} | "
        f"Recall OOF: {recall:.3f}"
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

    # --------------------------------------------------------
    # AUC
    # --------------------------------------------------------

    try:

        auc = roc_auc_score(
            y_test,
            y_proba
        )

    except ValueError:

        auc = np.nan

    # --------------------------------------------------------
    # Predicción utilizando el umbral OOF
    # --------------------------------------------------------

    y_pred = (
        y_proba >= umbral
    ).astype(int)

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    # --------------------------------------------------------
    # Matriz de confusión
    # --------------------------------------------------------

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
# UNA REPETICIÓN COMPLETA
# ============================================================
#
# Aquí está el cambio principal:
#
#     StratifiedGroupKFold
#
# se reemplaza por:
#
#     GroupShuffleSplit
#
# ============================================================

def correr_una_repeticion(
    X,
    y,
    groups,
    repeat_idx,
    random_state_repeat
):

    # --------------------------------------------------------
    # GroupShuffleSplit
    # --------------------------------------------------------
    #
    # La separación se realiza por grupo.
    #
    # Por lo tanto, todas las filas pertenecientes a un
    # mismo id_lote permanecen juntas.
    #
    # --------------------------------------------------------

    group_split = GroupShuffleSplit(

        n_splits=1,

        test_size=TEST_SIZE,

        random_state=random_state_repeat
    )

    resultados_repeticion = []

    # --------------------------------------------------------
    # Obtener una separación
    # --------------------------------------------------------

    train_idx, test_idx = next(
        group_split.split(
            X,
            y,
            groups
        )
    )

    # --------------------------------------------------------
    # Datos TRAIN
    # --------------------------------------------------------

    X_train = (
        X.iloc[train_idx]
        .copy()
    )

    y_train = (
        y.iloc[train_idx]
        .copy()
    )

    groups_train = (
        groups.iloc[train_idx]
        .copy()
    )

    # --------------------------------------------------------
    # Datos TEST
    # --------------------------------------------------------

    X_test = (
        X.iloc[test_idx]
        .copy()
    )

    y_test = (
        y.iloc[test_idx]
        .copy()
    )

    groups_test = (
        groups.iloc[test_idx]
        .copy()
    )

    # ========================================================
    # INFORMACIÓN DE LA PARTICIÓN
    # ========================================================

    print(
        f"\n   Rep {repeat_idx}/{N_SPLITS_EXTERNOS}"
    )

    print(
        "   ----------------------------------------"
    )

    print(
        f"   Train: {len(train_idx)} filas | "
        f"{groups_train.nunique()} lotes"
    )

    print(
        f"   Test:  {len(test_idx)} filas | "
        f"{groups_test.nunique()} lotes"
    )

    print(
        f"   Positivos train: "
        f"{y_train.mean():.3f}"
    )

    print(
        f"   Positivos test:  "
        f"{y_test.mean():.3f}"
    )

    # ========================================================
    # COMPROBAR QUE NO EXISTE INTERSECCIÓN DE LOTES
    # ========================================================

    lotes_train = set(
        groups_train.unique()
    )

    lotes_test = set(
        groups_test.unique()
    )

    interseccion = (
        lotes_train
        .intersection(
            lotes_test
        )
    )

    print(
        f"   Lotes compartidos: "
        f"{len(interseccion)}"
    )

    if len(interseccion) > 0:

        raise RuntimeError(
            "❌ ERROR: existen lotes "
            "compartidos entre TRAIN y TEST."
        )

    print(
        "   ✅ No existe fuga de lotes."
    )

    # ========================================================
    # OBTENER UMBRAL MEDIANTE OOF
    # ========================================================

    umbral = obtener_umbral_oof(
        X_train,
        y_train,
        groups_train,
        random_state_repeat
    )

    # ========================================================
    # PREPARAR FEATURES
    # ========================================================

    X_train = preparar_nulos(
        X_train
    )

    X_test = preparar_nulos(
        X_test
    )

    # ========================================================
    # ENCODERS
    # ========================================================

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

    # ========================================================
    # ENTRENAR RANDOM FOREST
    # ========================================================

    model = crear_modelo()

    model.fit(
        X_train_encoded,
        y_train
    )

    # ========================================================
    # PREDICCIÓN
    # ========================================================

    y_proba = (
        model
        .predict_proba(
            X_test_encoded
        )[:, 1]
    )

    # ========================================================
    # EVALUACIÓN
    # ========================================================

    resultado = evaluar_test(
        y_test,
        y_proba,
        umbral
    )

    resultado[
        "repeticion"
    ] = repeat_idx

    resultado[
        "fold"
    ] = repeat_idx

    resultado[
        "lotes_train"
    ] = groups_train.nunique()

    resultado[
        "lotes_test"
    ] = groups_test.nunique()

    resultado[
        "filas_train"
    ] = len(train_idx)

    resultado[
        "filas_test"
    ] = len(test_idx)

    resultados_repeticion.append(
        resultado
    )

    # ========================================================
    # MOSTRAR RESULTADOS
    # ========================================================

    print(
        f"\n   RESULTADO REPETICIÓN {repeat_idx}"
    )

    print(
        f"   AUC     = "
        f"{resultado['auc']:.3f}"
    )

    print(
        f"   Precision = "
        f"{resultado['precision']:.3f}"
    )

    print(
        f"   Recall    = "
        f"{resultado['recall']:.3f}"
    )

    print(
        f"   F1        = "
        f"{resultado['f1']:.3f}"
    )

    print(
        f"   Umbral    = "
        f"{resultado['umbral']:.2f}"
    )

    print(
        f"   TN={resultado['tn']} | "
        f"FP={resultado['fp']} | "
        f"FN={resultado['fn']} | "
        f"TP={resultado['tp']}"
    )

    return resultados_repeticion


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

        print(
            "🔌 Conexión cerrada."
        )

    # ========================================================
    # ELIMINAR FILAS SIN TARGET
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
    # GRUPOS
    # ========================================================
    #
    # id_lote NO es una feature.
    #
    # Se utiliza exclusivamente para realizar la separación
    # por lote.
    #
    # ========================================================

    groups = (
        df[
            "id_lote"
        ]
        .copy()
    )

    print(
        f"ℹ️ Lotes totales: "
        f"{groups.nunique()}"
    )

    # ========================================================
    # DISTRIBUCIÓN GENERAL DEL TARGET
    # ========================================================

    print(
        "\n📊 DISTRIBUCIÓN GLOBAL DEL TARGET"
    )

    print(
        f"   Negativos: "
        f"{(y := df[TARGET].astype(int)).value_counts().get(0, 0)}"
    )

    print(
        f"   Positivos: "
        f"{y.value_counts().get(1, 0)}"
    )

    print(
        f"   Tasa positiva: "
        f"{y.mean():.3f}"
    )

    # ========================================================
    # X / Y
    # ========================================================

    X = (
        df[
            FEATURES
        ]
        .copy()
    )

    y = (
        df[
            TARGET
        ]
        .astype(int)
    )

    print("\n")

    print(
        "=" * 70
    )

    print(
        "🚀 RANDOM FOREST — "
        "GROUPSHUFFLESPLIT"
    )

    print(
        "=" * 70
    )

    print(
        f"\n🎯 Target utilizado: "
        f"{TARGET}"
    )

    print(
        "📅 Ventana del target: "
        "4 semanas"
    )

    print(
        f"🔀 Separación externa: "
        f"GroupShuffleSplit"
    )

    print(
        f"📊 Test externo: "
        f"{TEST_SIZE * 100:.0f}% de los lotes"
    )

    print(
        f"🔁 Repeticiones: "
        f"{N_SPLITS_EXTERNOS}"
    )

    print(
        f"🔬 Validación interna: "
        f"StratifiedGroupKFold "
        f"({N_SPLITS_INTERNOS} folds)"
    )

    print(
        f"🎯 Recall mínimo: "
        f"{RECALL_MINIMO:.2f}"
    )

    todos_los_resultados = []

    # ========================================================
    # REPETICIONES
    # ========================================================

    for repeat_idx in range(
        1,
        N_SPLITS_EXTERNOS + 1
    ):

        random_state_repeat = (
            RANDOM_STATE
            + repeat_idx
        )

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"🔄 REPETICIÓN "
            f"{repeat_idx}/"
            f"{N_SPLITS_EXTERNOS}"
        )

        print(
            f"   Seed: "
            f"{random_state_repeat}"
        )

        print(
            "=" * 70
        )

        resultados_repeticion = (
            correr_una_repeticion(
                X,
                y,
                groups,
                repeat_idx,
                random_state_repeat
            )
        )

        todos_los_resultados.extend(
            resultados_repeticion
        )

    # ========================================================
    # RESULTADOS GLOBALES
    # ========================================================

    print("\n")

    print(
        "=" * 70
    )

    print(
        f"📊 RESULTADOS GLOBALES "
        f"({len(todos_los_resultados)} "
        f"separaciones)"
    )

    print(
        "=" * 70
    )

    df_resultados = pd.DataFrame(
        todos_los_resultados
    )

    # --------------------------------------------------------
    # AUC
    # --------------------------------------------------------

    auc_vals = (
        df_resultados[
            "auc"
        ]
        .dropna()
        .values
    )

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision_vals = (
        df_resultados[
            "precision"
        ]
        .values
    )

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    recall_vals = (
        df_resultados[
            "recall"
        ]
        .values
    )

    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    f1_vals = (
        df_resultados[
            "f1"
        ]
        .values
    )

    # ========================================================
    # IMPRIMIR AUC
    # ========================================================

    if len(auc_vals) > 0:

        print(
            f"\n   AUC-ROC:"
        )

        print(
            f"      Media = "
            f"{np.mean(auc_vals):.3f}"
        )

        print(
            f"      STD   = "
            f"{np.std(auc_vals):.3f}"
        )

        print(
            f"      Min   = "
            f"{np.min(auc_vals):.3f}"
        )

        print(
            f"      Max   = "
            f"{np.max(auc_vals):.3f}"
        )

    else:

        print(
            "\n   AUC-ROC: no disponible"
        )

    # ========================================================
    # RESTO DE MÉTRICAS
    # ========================================================

    print(
        f"\n   Precision:"
    )

    print(
        f"      Media = "
        f"{np.mean(precision_vals):.3f}"
    )

    print(
        f"      STD   = "
        f"{np.std(precision_vals):.3f}"
    )

    print(
        f"\n   Recall:"
    )

    print(
        f"      Media = "
        f"{np.mean(recall_vals):.3f}"
    )

    print(
        f"      STD   = "
        f"{np.std(recall_vals):.3f}"
    )

    print(
        f"\n   F1:"
    )

    print(
        f"      Media = "
        f"{np.mean(f1_vals):.3f}"
    )

    print(
        f"      STD   = "
        f"{np.std(f1_vals):.3f}"
    )

    # ========================================================
    # PROMEDIO POR REPETICIÓN
    # ========================================================

    print("\n")

    print(
        "=" * 70
    )

    print(
        "📋 RESULTADOS POR REPETICIÓN"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"{'Rep.':<8}"
        f"{'AUC':>10}"
        f"{'Prec.':>10}"
        f"{'Recall':>10}"
        f"{'F1':>10}"
        f"{'Umbral':>10}"
    )

    print(
        "-" * 58
    )

    for repeat_idx, grupo in (
        df_resultados
        .groupby(
            "repeticion"
        )
    ):

        print(
            f"{repeat_idx:<8}"
            f"{grupo['auc'].mean():>10.3f}"
            f"{grupo['precision'].mean():>10.3f}"
            f"{grupo['recall'].mean():>10.3f}"
            f"{grupo['f1'].mean():>10.3f}"
            f"{grupo['umbral'].mean():>10.2f}"
        )

    # ========================================================
    # RESUMEN DE SEPARACIONES
    # ========================================================

    print("\n")

    print(
        "=" * 70
    )

    print(
        "📋 RESUMEN DE LAS SEPARACIONES"
    )

    print(
        "=" * 70
    )

    print()

    for _, resultado in (
        df_resultados
        .iterrows()
    ):

        print(
            f"Rep {int(resultado['repeticion'])}: "
            f"Train={int(resultado['lotes_train'])} lotes | "
            f"Test={int(resultado['lotes_test'])} lotes | "
            f"AUC={resultado['auc']:.3f}"
        )

    # ========================================================
    # COMPARACIÓN
    # ========================================================

    print("\n")

    print(
        "=" * 70
    )

    print(
        "⚠️ COMPARACIÓN"
    )

    print(
        "=" * 70
    )

    print(
        "\nEsta versión utiliza:"
    )

    print(
        "   • GroupShuffleSplit para "
        "la evaluación externa."
    )

    print(
        "   • StratifiedGroupKFold para "
        "la selección OOF del umbral."
    )

    print(
        "   • id_lote exclusivamente "
        "como variable de agrupamiento."
    )

    print(
        "   • Random Forest para "
        "el modelo predictivo."
    )

    print(
        "\nLa comparación con otros modelos debe hacerse "
        "utilizando exactamente las mismas separaciones "
        "externas cuando sea posible."
    )

    # ========================================================
    # ENTRENAMIENTO DEL MODELO FINAL
    # ========================================================
    #
    # Una vez terminada la evaluación, se entrena el modelo
    # definitivo utilizando todos los lotes disponibles.
    #
    # El test externo NO se utiliza aquí porque las
    # evaluaciones anteriores ya fueron independientes.
    #
    # ========================================================

    print("\n")

    print(
        "=" * 70
    )

    print(
        "🚀 ENTRENANDO MODELO FINAL "
        "(Random Forest — 4 semanas)"
    )

    print(
        "=" * 70
    )

    X_final = (
        preparar_nulos(
            X
        )
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
    # GUARDAR MODELO
    # ========================================================

    os.makedirs(
        "models",
        exist_ok=True
    )

    # --------------------------------------------------------
    # Guardar modelo
    # --------------------------------------------------------

    with open(
        MODEL_PATH,
        "wb"
    ) as f:

        pickle.dump(
            modelo_final,
            f
        )

    # --------------------------------------------------------
    # Guardar encoders
    # --------------------------------------------------------

    with open(
        ENCODERS_PATH,
        "wb"
    ) as f:

        pickle.dump(
            encoders_final,
            f
        )

    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n✅ Modelo final "
        "(Random Forest) "
        "entrenado correctamente."
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

    print(
        "=" * 70
    )

    print(
        "✅ ENTRENAMIENTO FINALIZADO"
    )

    print(
        "=" * 70
    )

    return modelo_final


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    entrenar_modelo()