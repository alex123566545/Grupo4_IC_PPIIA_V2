
from src.config.database import get_connection
from src.training.train import train_model

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def mejor_umbral(y_true, y_proba, recall_minimo=0.60):

    candidatos = []

    for i in range(5, 100):

        umbral = i / 100

        y_pred = (y_proba >= umbral).astype(int)

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
                (umbral, precision, recall)
            )

    if not candidatos:

        recalls = []

        for i in range(5, 100):

            umbral = i / 100

            y_pred = (y_proba >= umbral).astype(int)

            recall = recall_score(
                y_true,
                y_pred,
                zero_division=0
            )

            recalls.append(
                (umbral, recall)
            )

        umbral, recall = max(
            recalls,
            key=lambda x: x[1]
        )

        y_pred = (y_proba >= umbral).astype(int)

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0
        )

        print(
            f"\n⚠️ Ningún umbral alcanzó "
            f"Recall >= {recall_minimo:.2f}."
        )

        print(
            f"   Se reporta el umbral con mayor "
            f"Recall disponible: {recall:.3f}"
        )

        return umbral, precision, recall

    umbral, precision, recall = max(
        candidatos,
        key=lambda x: x[1]
    )

    return umbral, precision, recall


def evaluar_modelo(model, X_test, y_test):

    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(
        y_test,
        y_proba
    )

    umbral, precision, recall = mejor_umbral(
        y_test,
        y_proba,
        recall_minimo=0.60
    )

    y_pred = (y_proba >= umbral).astype(int)

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred
    ).ravel()

    print("\n")
    print("=" * 60)
    print("📊 EVALUACIÓN DEL MODELO")
    print("=" * 60)

    print(
        "\n📌 Evaluación en hold-out "
        "(split aleatorio, 25%):"
    )

    print(f"   AUC-ROC:          {auc:.3f}")
    print(f"   Umbral elegido:   {umbral:.2f}")
    print(
        "   Criterio:         "
        "Recall >= 0.60, maximizando Precision"
    )
    print(f"   Precision:        {precision:.3f}")
    print(f"   Recall:           {recall:.3f}")
    print(f"   F1:               {f1:.3f}")
    print(
        f"   Matriz confusión: "
        f"TP={tp} FP={fp} FN={fn} TN={tn}"
    )

    print("\n⚠️ IMPORTANTE")

    print(
        "   Esta evaluación corresponde a un "
        "split aleatorio."
    )

    print(
        "   La validación más rigurosa del proyecto "
        "es GroupKFold por lote."
    )

    print(
        "   Un split aleatorio puede sobreestimar "
        "el desempeño real si existen registros "
        "relacionados entre entrenamiento y prueba."
    )

    print("=" * 60)


def main():

    print("🔌 Conectando a Supabase...")

    conn = get_connection()

    try:

        print(
            "🚀 Entrenando modelo desde "
            "gold_ml.dataset_features..."
        )

        model, X_test, y_test = train_model(conn)

        evaluar_modelo(
            model,
            X_test,
            y_test
        )

    finally:

        conn.close()

        print("\n🔌 Conexión cerrada.")


if __name__ == "__main__":
    main()
