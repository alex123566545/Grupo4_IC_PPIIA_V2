# ==========================================================
# DIAGNÓSTICO — feature_importances_ y comparación
#              lotes reales vs. lotes sintéticos
# ==========================================================
#
# Este script responde a dos preguntas concretas:
#
# 1. ¿Las variables que la bitácora del proyecto marcó como
#    "ruido" (lags, interacción desparasitación x animales
#    nuevos) siguen siendo poco importantes con el dataset
#    aumentado, o el FEATURES actual (sin depurar) está
#    metiendo ruido nuevo?
#
# 2. ¿La caída de AUC (0.81 -> 0.672 en split aleatorio) viene
#    de los lotes sintéticos que se agregaron, o es un problema
#    parejo en todo el dataset?
#
# No reemplaza a main.py: es un script aparte para diagnóstico,
# entrena su propio modelo (mismos hiperparámetros que train.py)
# para tener control total sobre el split y poder etiquetar cada
# fila de test como "real" o "sintética".
#

import os
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix

from src.config.settings import FEATURES, TARGET, TEST_SIZE, RANDOM_STATE
from src.config.database import get_connection

# Lotes reales originales del proyecto (documento histórico, Etapa 1-5).
# Todo lo que no esté en esta lista se considera lote sintético agregado
# en la etapa de aumento de datos.
LOTES_REALES = {f"LOTE_{i:02d}" for i in range(1, 9)}


# ==========================================================
# UMBRAL: recall >= 0.60 como piso, maximizando precision
# (mismo criterio que main.py, copiado aquí para que el script
# sea independiente y no dependa de reentrenar vía train_model)
# ==========================================================

def mejor_umbral(y_true, y_proba, recall_minimo=0.60):
    candidatos = []
    for i in range(5, 100):
        umbral = i / 100
        y_pred = (y_proba >= umbral).astype(int)
        recall = recall_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        if recall >= recall_minimo:
            candidatos.append((umbral, precision, recall))

    if not candidatos:
        recalls = [
            (i / 100, recall_score(y_true, (y_proba >= i / 100).astype(int), zero_division=0))
            for i in range(5, 100)
        ]
        umbral, recall = max(recalls, key=lambda x: x[1])
        precision = precision_score(y_true, (y_proba >= umbral).astype(int), zero_division=0)
        return umbral, precision, recall

    umbral, precision, recall = max(candidatos, key=lambda x: x[1])
    return umbral, precision, recall


def imprimir_metricas(nombre, y_true, y_proba, umbral):
    if len(y_true) == 0:
        print(f"\n   {nombre}: sin filas en este subconjunto de test.")
        return
    if y_true.nunique() < 2:
        print(f"\n   {nombre}: solo hay una clase presente en este subconjunto ({len(y_true)} filas), no se puede calcular AUC.")
        return

    y_pred = (y_proba >= umbral).astype(int)
    auc = roc_auc_score(y_true, y_proba)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print(f"\n   {nombre}  (n={len(y_true)}, positivos={int(y_true.sum())})")
    print(f"      AUC-ROC:   {auc:.3f}")
    print(f"      Precision: {precision:.3f}   Recall: {recall:.3f}   F1: {f1:.3f}")
    print(f"      TP={tp} FP={fp} FN={fn} TN={tn}")


def main():
    print("🔌 Conectando a Supabase...")
    conn = get_connection()

    try:
        print("📥 Leyendo gold_ml.dataset_features...")
        # ORDER BY explícito: sin esto, train_test_split(random_state=...) no
        # es reproducible entre corridas si Postgres devuelve las filas en
        # distinto orden (no hay ORDER BY implícito garantizado en SQL).
        query = """
            SELECT *
            FROM gold_ml.dataset_features
            ORDER BY id_lote, fecha
        """
        df = pd.read_sql(query, conn)
    finally:
        conn.close()

    filas_antes = len(df)
    df = df[df[TARGET].notna()].copy()
    print(f"ℹ️  Filas excluidas por target nulo: {filas_antes - len(df)}")
    print(f"ℹ️  Filas totales para entrenar/evaluar: {len(df)}")
    print(f"ℹ️  Lotes: {df['id_lote'].nunique()} "
          f"({sum(1 for l in df['id_lote'].unique() if l in LOTES_REALES)} reales, "
          f"{sum(1 for l in df['id_lote'].unique() if l not in LOTES_REALES)} sintéticos)")

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)
    grupo_lote = df["id_lote"]

    # misma codificación categórica que train.py
    categorical_columns = X.select_dtypes(include=["object", "category"]).columns
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    X_train, X_test, y_train, y_test, lote_train, lote_test = train_test_split(
        X, y, grupo_lote,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\n🚀 Entrenando Random Forest (mismos hiperparámetros que train.py)...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # ======================================================
    # 1. feature_importances_
    # ======================================================
    print("\n" + "=" * 60)
    print("📌 IMPORTANCIA DE VARIABLES (feature_importances_)")
    print("=" * 60)

    importancias = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    for i, (feat, val) in enumerate(importancias.items(), start=1):
        marca = "  ⚠️ posible ruido" if val < 0.01 else ""
        print(f"   {i:2d}. {feat:<40s} {val:.4f}{marca}")

    print("\n   Compara esta tabla contra la Etapa 4.3 de la bitácora:")
    print("   las variables de rezago (lag1) y la interacción desparasitación x")
    print("   animales_nuevos ya habían salido con importancia casi nula ahí.")
    print("   Si vuelven a salir bajas acá, no fue casualidad del dataset viejo.")

    # ======================================================
    # 2. Evaluación general vs. real vs. sintético
    # ======================================================
    print("\n" + "=" * 60)
    print("📌 EVALUACIÓN: TODO EL TEST vs. SOLO LOTES REALES vs. SOLO SINTÉTICOS")
    print("=" * 60)

    y_proba_test = model.predict_proba(X_test)[:, 1]
    umbral, _, _ = mejor_umbral(y_test, y_proba_test, recall_minimo=0.60)
    print(f"\n   Umbral usado en las 3 evaluaciones (recall>=0.60, maximizando precision): {umbral:.2f}")

    es_real_test = lote_test.isin(LOTES_REALES)

    imprimir_metricas("TODO el conjunto de test", y_test, y_proba_test, umbral)
    imprimir_metricas("Solo lotes REALES (LOTE_01-08)", y_test[es_real_test], y_proba_test[es_real_test.values], umbral)
    imprimir_metricas("Solo lotes SINTÉTICOS (LOTE_09+)", y_test[~es_real_test], y_proba_test[~es_real_test.values], umbral)

    print("\n" + "=" * 60)
    print("Lectura sugerida:")
    print(" - Si el AUC de 'solo reales' se acerca a 0.81 (el valor histórico)")
    print("   y el de 'solo sintéticos' es notablemente menor, la caída viene")
    print("   de la generación de datos sintéticos, no del pipeline de entrenamiento.")
    print(" - Si ambos subconjuntos rinden parecido (y ambos bajos), el problema")
    print("   está en el FEATURES sin depurar o en otro cambio del pipeline.")
    print("=" * 60)


if __name__ == "__main__":
    main()