import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
import json
import matplotlib.pyplot as plt
from tcn import TCN

model = tf.keras.models.load_model(
    "C:/Users/lohit ramaraju/OneDrive/Desktop/IITJ/Main Project/saved_models/CNN_TCN_model.h5",
    custom_objects={'TCN': TCN}
)

with open("C:/Users/lohit ramaraju/OneDrive/Desktop/IITJ/Main Project/saved_models/class_indices_Fusion.json", 'r') as f:
    class_indices = json.load(f)
class_names = list(class_indices.keys())

def preprocess_image(img, img_size=(128, 128)):
    img_array = cv2.resize(img, img_size) 
    img_array = np.expand_dims(img_array, axis=0)   
    img_array = img_array / 255.0   
    return img_array

def predict_class(img_array):
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])  
    return class_names[predicted_class], predictions[0][predicted_class]  

def get_gradcam_heatmap(model, image, label):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[:, label]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap) 
    return heatmap.numpy()

def display_results(original_img, heatmap, overlay_img):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original_img)
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.title("Heatmap")
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(overlay_img)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

cap = cv2.VideoCapture(0)  

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    original_img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    preprocessed_image = preprocess_image(original_img_rgb)

    predicted_label, confidence = predict_class(preprocessed_image)
    print(f"Predicted Class: {predicted_label} with confidence {confidence:.2f}")
    heatmap = get_gradcam_heatmap(model, preprocessed_image, np.argmax(model.predict(preprocessed_image)[0]))
    heatmap = np.uint8(255 * heatmap)  
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  
    overlay_img = cv2.addWeighted(original_img_rgb, 0.4, heatmap, 0.6, 0)  
    display_results(original_img_rgb, heatmap, overlay_img)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("extracted_frame.jpg", frame) 
        print("Frame saved as 'extracted_frame.jpg'")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
