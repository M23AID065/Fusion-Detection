import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tcn import TCN  
import cv2  
import json
import matplotlib.pyplot as plt

model = tf.keras.models.load_model(
    "C:/Users/lohit ramaraju/OneDrive/Desktop/IITJ/Main Project/saved_models/TCN_model.h5",
    custom_objects={'TCN': TCN} 
)
with open("C:/Users/lohit ramaraju/OneDrive/Desktop/IITJ/Main Project/saved_models/class_indices.json", 'r') as f:
    class_indices = json.load(f)
class_names = list(class_indices.keys())
def preprocess_image(img_path, img_size=(128, 128)):
    img = image.load_img(img_path, target_size=img_size) 
    img_array = image.img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = img_array / 255.0  
    return img_array
def predict_class(img_array):
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)  
    return class_names[predicted_class], predictions[0][predicted_class]  
def get_gradcam_heatmap(model, image, label):
    from tensorflow.keras import layers
    last_conv_layer = None
    for layer in model.layers:
        if isinstance(layer, layers.Conv2D):  
            last_conv_layer = layer.name
    if last_conv_layer is None:
        raise ValueError("No Conv2D layer found in the CNN part of the model. Grad-CAM requires a convolutional layer.")
    grad_model = tf.keras.models.Model([model.inputs], 
                                       [model.get_layer(last_conv_layer).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[:, label]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
def display_gradcam(image, heatmap, alpha=0.4):
    img = image.squeeze()
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    output_img = cv2.addWeighted(img, alpha, heatmap, 1 - alpha, 0)
    plt.figure(figsize=(10, 10))
    plt.imshow(output_img)
    plt.axis('off')
    plt.show()
image_path = "C:/Users/lohit ramaraju/OneDrive/Desktop/IITJ/Main Project/Dataset/annotated_samples/101a5dc8b7a0104d99ee019a9930f8d9.jpg"
preprocessed_image = preprocess_image(image_path)
predicted_label, confidence = predict_class(preprocessed_image)
print(f"Predicted Class: {predicted_label} with confidence {confidence:.2f}")
heatmap = get_gradcam_heatmap(model, preprocessed_image, np.argmax(model.predict(preprocessed_image)))
display_gradcam(preprocessed_image.squeeze(), heatmap)
