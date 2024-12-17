import tensorflow as tf
from tensorflow.keras.models import load_model

# Caminho para o modelo
model_path = 'D:/Git/reconhecimento_facial_yolo/modelo_classificacao_faces.h5'

# Carregar o modelo de classificação
try:
    model = load_model(model_path)
    print("Modelo carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
