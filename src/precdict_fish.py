import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from pathlib import Path

# -- Configurações --
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = r"C:\Users\VISIONLAB\Desktop\clone tafresco model\tafrescomodel\models\tafresco_mvp_v1.keras"
IMG_SIZE = (224, 224)
CLASSES = ["Fresh", "Highly_Fresh", "Not_Fresh"] # Ordem alfabética padrão do Keras

def predict_fish_freshness(img_path):
    # 1. Carrega o modelo
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # 2. Carrega e prepara a imagem
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Cria o batch (1, 224, 224, 3)
    
    # 3. Pré-processamento (IGUAL ao usado no treino)
    img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    # 4. Predição
    predictions = model.predict(img_preprocessed)
    score = tf.nn.softmax(predictions[0]) # Normaliza as probabilidades
    
    # 5. Resultado
    class_idx = np.argmax(predictions[0])
    confidence = 100 * np.max(predictions[0])
    
    print(f"\n--- Resultado da Análise ---")
    print(f"Imagem: {Path(img_path).name}")
    print(f"Predição: {CLASSES[class_idx]}")
    print(f"Confiança: {confidence:.2f}%")
    print("----------------------------")

if __name__ == "__main__":
    # COLOQUE O CAMINHO DA SUA FOTO NOVA AQUI
    minha_foto = r"C:\Users\VISIONLAB\Desktop\olhos peixes\olhos opacos\ee5bdb59-166e-49ad-bb84-a089fdcb0a42.jpg"
    predict_fish_freshness(minha_foto)