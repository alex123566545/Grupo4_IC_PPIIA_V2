from src.training.randomSearch import entrenar_modelo


def main():

    print("=" * 70)
    print("🐄 SIPREM-BOVINO")
    print("🔎 RANDOMIZED SEARCH + GROUPKFOLD")
    print("=" * 70)

    entrenar_modelo()

    print("\n" + "=" * 70)
    print("✅ PROCESO FINALIZADO")
    print("=" * 70)


if __name__ == "__main__":
    main()
