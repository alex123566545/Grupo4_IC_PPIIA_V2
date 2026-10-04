"""
SIPREM-BOVINO
Punto de entrada para ejecutar el (re)entrenamiento del modelo.

Uso:
    python main.py
"""

import sys

from src.training.train2 import entrenar_modelo


def main():
    try:
        entrenar_modelo()
    except Exception as e:
        print(f"\n❌ El entrenamiento falló: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()