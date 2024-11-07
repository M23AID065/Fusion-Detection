import pathlib
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers, models, losses, optimizers
from tcn import TCN  
import json
import matplotlib.pyplot as plt

FILE_DIR = "C:/Users/lohit ramaraju/OneDrive/Desktop/IITJ/Main Project/Dataset/crop/"
DATA_PATH = pathlib.Path(FILE_DIR)
IMG_SIZE = (128, 128)
BATCH_SIZE = 32        
EPOCHS = 10 
datagen = image.ImageDataGenerator(rescale=1./255, validation_split=0.3)
train_generator = datagen.flow_from_directory(
    FILE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse', 
    subset='training'
)
val_generator = datagen.flow_from_directory(
    FILE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='validation'
)
nClass = len(train_generator.class_indices)
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
])
cnn_output_shape = model.output_shape
print("CNN output shape after MaxPooling:", cnn_output_shape)
model.add(layers.Reshape((cnn_output_shape[1] * cnn_output_shape[2], cnn_output_shape[3])))
model.add(TCN(64, return_sequences=False))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(nClass, activation='softmax'))
model.compile(optimizer='adam', 
                loss=losses.SparseCategoricalCrossentropy(), 
                metrics=['accuracy'])
model.summary()
history = model.fit(train_generator, 
                    validation_data=val_generator, 
                    epochs=EPOCHS)
model.save("C:/Users/lohit ramaraju/OneDrive/Desktop/IITJ/Main Project/saved_models/CNN_TCN_model.h5")
with open("C:/Users/lohit ramaraju/OneDrive/Desktop/IITJ/Main Project/saved_models/class_indices_Fusion.json", 'w') as f:
    json.dump(train_generator.class_indices, f)

print("Model and class indices saved successfully.")
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
plot_training_history(history)
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
