# ============================================================
# SIPREM-BOVINO
# MAIN — ENTRENAMIENTO RANDOM FOREST
# SPLIT ALEATORIO POR LOTE
# ============================================================

from src.training.splitAleatorio import entrenar_modelo


def main():

    print("\n")
    print("=" * 70)
    print("🐄 SIPREM-BOVINO")
    print("🌲 RANDOM FOREST — SPLIT ALEATORIO POR LOTE")
    print("=" * 70)

    try:

        entrenar_modelo()

    except Exception as e:

        print("\n")
        print("=" * 70)
        print("❌ ERROR DURANTE EL ENTRENAMIENTO")
        print("=" * 70)

        print(f"\n{type(e).__name__}: {e}")

        raise

    print("\n")
    print("=" * 70)
    print("🏁 PROCESO FINALIZADO")
    print("=" * 70)


if __name__ == "__main__":

    main()