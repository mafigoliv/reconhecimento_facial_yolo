import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Caminhos para os arquivos cfg, weights, modelo e a imagem
cfg_path = 'D:/Git/reconhecimento_facial_yolo/yolov3.cfg'
weights_path = 'D:/Git/reconhecimento_facial_yolo/yolov3.weights'
model_path = 'D:/Git/reconhecimento_facial_yolo/modelo_classificacao_faces.h5'
image_dir = 'D:/Git/reconhecimento_facial_yolo/train/images/characters/series/'  # Caminho da pasta com as imagens
label_dir = 'D:/Git/reconhecimento_facial_yolo/train/labels/characters/series/'  # Caminho da pasta com os arquivos de coordenadas
output_dir = 'D:/Git/reconhecimento_facial_yolo/output/'  # Caminho para salvar as imagens processadas

# Verificar se os arquivos existem
print(f"Arquivo de configuração existe: {os.path.exists(cfg_path)}")
print(f"Arquivo de pesos existe: {os.path.exists(weights_path)}")
print(f"Arquivo do modelo de classificação existe: {os.path.exists(model_path)}")

# Crie a pasta de saída, se não existir
os.makedirs(output_dir, exist_ok=True)

# Carregar o modelo YOLOv3 pré-treinado para detecção de faces
net = cv2.dnn.readNet(weights_path, cfg_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Carregar o modelo de classificação
model = load_model(model_path)
print("Modelo de classificação carregado com sucesso!")

# Dicionário de mapeamento de IDs para nomes de personagens
character_names = {
    0: "Walter White",
    1: "Jesse Pinkman",
    2: "Skyler White",
    3: "Gustavo Fring",
    4: "Saul Goodman",
    5: "Hank Schrader",
    6: "Marie Schrader",
    7: "Mike Ehrmantraut"
    # Adicione outros personagens conforme necessário
}

def detect_and_recognize_face(img, labels):
    height, width, channels = img.shape

    # Processar cada linha de coordenada
    for label in labels:
        class_id, x_center, y_center, w, h = map(float, label.split())
        x_center *= width
        y_center *= height
        w *= width
        h *= height
        x = int(x_center - w / 2)
        y = int(y_center - h / 2)
        w = int(w)
        h = int(h)

        # Garantir que as coordenadas estejam dentro dos limites da imagem
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(0, min(w, width - x))
        h = max(0, min(h, height - y))

        # Desenhar a caixa delimitadora 
        color = (255, 255, 255)  
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        # Extrair a face e redimensioná-la para o modelo de classificação
        face = img[y:y + h, x:x + w]
        face = cv2.resize(face, (224, 224))
        face = face.astype("float") / 255.0
        face = image.img_to_array(face)
        face = np.expand_dims(face, axis=0)

        # Prever a classe da face usando o modelo treinado
        prediction = model.predict(face)

        # Obter a classe com maior probabilidade
        class_index = np.argmax(prediction[0])
        confidence = prediction[0][class_index]

        # Mostrar a classe e a confiança na imagem com ajustes
        character_name = character_names.get(int(class_id), f"ID: {int(class_id)}")
        label = f"{character_name} ({confidence * 100:.2f})"
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)  
        cv2.rectangle(img, (x, y - label_height - 10), (x + label_width, y), (0, 138, 0), cv2.FILLED)
        cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return img

def main():
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            # Caminho completo do arquivo de imagem e coordenadas
            image_path = os.path.join(image_dir, filename)
            label_path = os.path.join(label_dir, filename.replace('.jpg', '.txt'))

            if not os.path.exists(label_path):
                print(f"Arquivo de coordenadas não encontrado para {image_path}")
                continue

            # Carregar a imagem
            img = cv2.imread(image_path)
            if img is None:
                print(f"Erro ao carregar a imagem {image_path}.")
                continue

            # Carregar as coordenadas das faces
            with open(label_path, 'r') as file:
                labels = file.readlines()

            # Detectar e reconhecer faces na imagem
            img = detect_and_recognize_face(img, labels)

            # Verificar se o arquivo de saída já existe e alterar o nome se necessário
            output_path = os.path.join(output_dir, filename)
            base, ext = os.path.splitext(output_path)
            i = 1
            while os.path.exists(output_path):
                output_path = f"{base}_{i}{ext}"
                i += 1

            # Salvar a imagem processada
            cv2.imwrite(output_path, img)
            print(f"Imagem processada salva em {output_path}")

            # Redimensionar a janela para se ajustar à imagem
            cv2.namedWindow('Face Recognition', cv2.WINDOW_NORMAL)
            cv2.imshow('Face Recognition', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
