# ============================================================
# SIPREM-BOVINO
# MAIN — EJECUCIÓN DE PREDICCIONES
# ============================================================

from src.data.prediccion import realizar_prediccion


def main():

    print("\n" + "=" * 70)
    print("🐄 SIPREM-BOVINO")
    print("🤖 SISTEMA DE PREDICCIÓN")
    print("=" * 70)

    try:

        resultado = realizar_prediccion()

        print(
            f"\n✅ Se generaron "
            f"{len(resultado)} predicciones correctamente."
        )

    except Exception as e:

        print(
            "\n❌ ERROR DURANTE LA PREDICCIÓN:"
        )

        print(
            f"   {e}"
        )

        raise


if __name__ == "__main__":

    main()