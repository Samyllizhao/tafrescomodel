"""
Avaliação do Modelo - TáFresco
Gera matriz de confusão e relatório de classificação.
"""
from pathlib import Path

import tensorflow as tf
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# ── Configurações ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models" / "tafresco_mvp_v1.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_NAMES = ["Fresh", "Highly_Fresh", "Not_Fresh"]


def evaluate():
    """Carrega modelo, faz previsões no val set e gera métricas."""
    # 1. Dataset de validação
    print("1. Carregando dataset de validação...")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        str(DATA_DIR / "val"),
        shuffle=False,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # 2. Carregar modelo
    print("\n2. Carregando modelo salvo...")
    model = tf.keras.models.load_model(str(MODEL_PATH))

    # 3. Coletar previsões
    print("\n3. Gerando previsões...")
    y_true = []
    y_pred_probs = []

    for images, labels in val_ds:
        y_true.extend(labels.numpy())
        preds = model.predict(images, verbose=0)
        y_pred_probs.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 4. Matriz de confusão
    print("\n4. Gerando Matriz de Confusão...")
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.title("Matriz de Confusão - TáFresco MVP", pad=20, fontsize=14)
    plt.ylabel("Rótulo Real", fontsize=12)
    plt.xlabel("Previsão do Modelo", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(str(PROJECT_ROOT / "models" / "confusion_matrix.png"), dpi=150)
    plt.show()

    # 5. Relatório
    print("\n>> Relatorio de Classificacao:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))


if __name__ == "__main__":
    evaluate()
