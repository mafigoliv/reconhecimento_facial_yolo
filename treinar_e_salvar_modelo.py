import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Diretórios dos dados
train_dir = 'D:/Git/reconhecimento_facial_yolo/train/images/characters'
val_dir = 'D:/Git/reconhecimento_facial_yolo/val/images/characters'

# Configuração do gerador de dados de treinamento e validação
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='sparse')
val_generator = val_datagen.flow_from_directory(val_dir, target_size=(224, 224), batch_size=32, class_mode='sparse')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Construir o modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')  # O número de classes do seu conjunto de dados
])

# Compilar o modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
model.fit(train_generator, epochs=10, validation_data=val_generator)

# Salvar o modelo
model.save('D:/Git/reconhecimento_facial_yolo/modelo_classificacao_faces.h5')
