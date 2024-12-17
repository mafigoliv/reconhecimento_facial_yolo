import h5py
from keras.models import load_model

# Caminho do arquivo do modelo
model_path = 'D:/Git/reconhecimento_facial_yolo/modelo_classificacao_faces.h5'

try:
    # Verificar a integridade do arquivo
    with h5py.File(model_path, 'r') as f:
        print("O arquivo abriu corretamente.")

    # Carregar o modelo
    model = load_model(model_path)

    # Imprimir um resumo do modelo
    model.summary()
    
except OSError as e:
    print("Erro ao abrir o arquivo:", e)
except Exception as e:
    print("Erro ao carregar o modelo:", e)
