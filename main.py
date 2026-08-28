import sys


def main():

    modelo = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "rf"
    )

    print("=" * 70)
    print("🐄 SIPREM-BOVINO")
    print("🎯 TARGET: RIESGO ALTO — 4 SEMANAS")
    print("=" * 70)

    # ========================================================
    # HISTGRADIENTBOOSTING
    # ========================================================

    if modelo == "hgb":

        print(
            "🚀 ENTRENAMIENTO Y VALIDACIÓN — "
            "HISTGRADIENTBOOSTING"
        )

        print("=" * 70)

        from src.training.train_hgb import (
            entrenar_modelo_hgb
        )

        entrenar_modelo_hgb()

    # ========================================================
    # RANDOM FOREST
    # ========================================================

    elif modelo == "rf":

        print(
            "🚀 ENTRENAMIENTO Y VALIDACIÓN — "
            "RANDOM FOREST"
        )

        print("=" * 70)

        from src.training.train import (
            entrenar_modelo
        )

        entrenar_modelo()

    # ========================================================
    # MODELO DESCONOCIDO
    # ========================================================

    else:

        print(
            f"❌ Modelo desconocido: '{modelo}'. "
            "Usa 'rf' o 'hgb'."
        )

        sys.exit(1)

    # ========================================================
    # FINALIZACIÓN
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "✅ PROCESO FINALIZADO"
    )

    print("=" * 70)


if __name__ == "__main__":

    main()