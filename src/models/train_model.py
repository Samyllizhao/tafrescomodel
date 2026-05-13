"""
Treinamento do Modelo - TáFresco
Transfer Learning com MobileNetV2 (base congelada).
"""
import os
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt

# ── Configurações ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
IMG_SIZE = (224, 224)   # tamanho nativo ideal para MobileNetV2
NUM_CLASSES = 3
EPOCHS = 25
MODEL_PATH = MODELS_DIR / "tafresco_mvp_v1.keras"


def load_datasets():
    """Carrega os datasets de treino e validação."""
    print("Carregando datasets...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        str(DATA_DIR / "train"),
        shuffle=True, 
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        str(DATA_DIR / "val"),
        shuffle=False,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    class_names = train_ds.class_names
    print(f"Classes identificadas: {class_names}")

    # otimização de performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names


def build_model():
    """Constrói o modelo MobileNetV2 com transfer learning."""
    # data augmentation (simulando condições da feira)
    data_augmentation = tf.keras.Sequential([
        layers.RandomRotation(0.15),
        layers.RandomBrightness(factor=0.2),
        layers.RandomFlip("horizontal_and_vertical"),
    ], name="camada_aumento_dados")

    # base pré-treinada
    print("Baixando pesos da MobileNetV2...")
    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # congela a base

    # cabeça de classificação
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(model, train_ds, val_ds):
    """Treina o modelo com EarlyStopping e ModelCheckpoint."""
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print(f"\n>> Iniciando treinamento para ate {EPOCHS} epocas...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )
    return history


def plot_history(history):
    """Gera gráficos de acurácia e loss."""
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Acurácia Treino", linewidth=2)
    plt.plot(val_acc, label="Acurácia Validação", linewidth=2, linestyle="--")
    plt.title("Evolução da Acurácia")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Erro (Loss) Treino", linewidth=2)
    plt.plot(val_loss, label="Erro (Loss) Validação", linewidth=2, linestyle="--")
    plt.title("Evolução do Erro")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(PROJECT_ROOT / "models" / "training_curves.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    train_ds, val_ds, class_names = load_datasets()
    model = build_model()
    model.summary()

    history = train(model, train_ds, val_ds)
    plot_history(history)

    print(f"\n>> Treinamento finalizado. Modelo salvo em: {MODEL_PATH}")
