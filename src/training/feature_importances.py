import os
import pickle

import pandas as pd
import matplotlib.pyplot as plt

from src.config.settings import (
    FEATURES,
    MODEL_PATH,
    ENCODERS_PATH
)


def analizar_importancias(top_n=None, umbral_ruido=None):
    """
    Carga el modelo final entrenado (models/model.pkl) y reporta la
    importancia de cada variable, siguiendo la misma metodología de la
    Etapa 4 del proyecto: no se descarta nada "a ojo", se decide con
    feature_importances_ como evidencia.

    Parámetros
    ----------
    top_n : int, opcional
        Si se indica, solo se muestran las top_n variables más importantes.
    umbral_ruido : float, opcional
        Si se indica, se listan aparte las variables con importancia por
        debajo de este valor, como candidatas a descartar (igual que
        casos_respiratorios_lag1 / casos_diarreicos_lag1 en la Etapa 4).
        Si no se indica, se usa 1 / (número de features) / 5 como
        referencia (una variable "sin señal" debería aportar bastante
        menos que el promedio si todas pesaran igual).

    Retorna
    -------
    df_importancias : DataFrame
        Columnas: feature, importancia, importancia_acumulada, ranking.
    """

    # ======================================================
    # CARGA DEL MODELO ENTRENADO
    # ======================================================

    print(f"📂 Cargando modelo desde {MODEL_PATH}...")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    if not hasattr(model, "feature_importances_"):
        raise ValueError(
            "❌ El modelo cargado no tiene feature_importances_ "
            "(¿es un RandomForestClassifier / modelo basado en árboles?)."
        )

    importancias = model.feature_importances_

    if len(importancias) != len(FEATURES):
        raise ValueError(
            f"❌ El modelo tiene {len(importancias)} features pero "
            f"FEATURES en settings.py tiene {len(FEATURES)}. "
            "El modelo guardado no corresponde a la lista actual de "
            "FEATURES -- vuelve a entrenar antes de analizar importancias."
        )

    # ======================================================
    # ARMAR TABLA ORDENADA DE IMPORTANCIAS
    # ======================================================

    df_importancias = pd.DataFrame({
        "feature": FEATURES,
        "importancia": importancias
    }).sort_values("importancia", ascending=False).reset_index(drop=True)

    df_importancias["ranking"] = df_importancias.index + 1

    df_importancias["importancia_acumulada"] = df_importancias["importancia"].cumsum()

    if umbral_ruido is None:
        umbral_ruido = (1 / len(FEATURES)) / 5

    df_importancias["candidata_a_descartar"] = df_importancias["importancia"] < umbral_ruido

    # ======================================================
    # REPORTE EN CONSOLA
    # ======================================================

    tabla = df_importancias if top_n is None else df_importancias.head(top_n)

    print("\n" + "=" * 70)
    print("📊 IMPORTANCIA DE VARIABLES (Random Forest)")
    print("=" * 70)

    for _, fila in tabla.iterrows():
        marca = "⚠️ " if fila["candidata_a_descartar"] else "   "
        print(
            f"{marca}{fila['ranking']:>2}. "
            f"{fila['feature']:<38} "
            f"{fila['importancia']:.4f}  "
            f"(acum. {fila['importancia_acumulada']:.3f})"
        )

    descartables = df_importancias[df_importancias["candidata_a_descartar"]]

    print("\n" + "-" * 70)
    print(f"ℹ️  Umbral de ruido usado: {umbral_ruido:.4f}")
    print(f"ℹ️  Variables candidatas a descartar (importancia casi nula): {len(descartables)}")

    if len(descartables) > 0:
        print("\n   " + ", ".join(descartables["feature"].tolist()))
        print(
            "\n   ⚠️  No las quites de FEATURES todavía sin verificar: "
            "vuelve a entrenar y correr GroupKFold sin ellas, y compara "
            "el AUC/F1 promedio contra el actual (mismo criterio que la "
            "Etapa 4 -- se descarta solo si el desempeño honesto no cae)."
        )

    print("=" * 70)

    # ======================================================
    # GUARDAR RESULTADOS (CSV + gráfico)
    # ======================================================

    os.makedirs("models", exist_ok=True)

    ruta_csv = "models/feature_importances.csv"
    df_importancias.to_csv(ruta_csv, index=False)
    print(f"\n💾 Tabla guardada en {ruta_csv}")

    ruta_png = "models/feature_importances.png"

    fig, ax = plt.subplots(figsize=(9, max(6, len(FEATURES) * 0.28)))

    grafico = df_importancias.sort_values("importancia", ascending=True)

    colores = ["#D85A30" if c else "#0F6E56" for c in grafico["candidata_a_descartar"]]

    ax.barh(grafico["feature"], grafico["importancia"], color=colores)
    ax.set_xlabel("Importancia (Random Forest)")
    ax.set_title("Importancia de variables — SIPREM-BOVINO")
    ax.axvline(umbral_ruido, color="grey", linestyle="--", linewidth=1, label=f"Umbral de ruido ({umbral_ruido:.4f})")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(ruta_png, dpi=150)
    plt.close(fig)

    print(f"💾 Gráfico guardado en {ruta_png}")

    return df_importancias


if __name__ == "__main__":

    analizar_importancias()