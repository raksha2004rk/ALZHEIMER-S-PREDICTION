import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# ---------------- LOAD MODEL ---------------- #
MODEL_PATH = "artifacts/model.h5"

if not os.path.exists(MODEL_PATH):
    raise Exception("Model not found! Train model first.")

model = tf.keras.models.load_model(MODEL_PATH)

# ---------------- IMAGE PATH ---------------- #
IMAGE_PATH = input("Enter path of MRI image: ")

if not os.path.exists(IMAGE_PATH):
    raise Exception("Image not found!")

# ---------------- LOAD IMAGE ---------------- #
image = cv2.imread(IMAGE_PATH)
image_resized = cv2.resize(image, (128, 128))
image_norm = image_resized / 255.0
input_image = np.expand_dims(image_norm, axis=0)

# ---------------- PREDICTION ---------------- #
prediction = model.predict(input_image)
class_idx = np.argmax(prediction)
confidence = np.max(prediction) * 100

classes = ["Non-Demented", "Moderate Demented"]
result = classes[class_idx]

print("\n===== RESULT =====")
print("Prediction:", result)
print(f"Confidence: {confidence:.2f}%")

# ---------------- GRAD-CAM ---------------- #
def get_gradcam(model, image):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer("last_conv").output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(image, heatmap):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

heatmap = get_gradcam(model, input_image)
gradcam_image = overlay_heatmap(image_resized, heatmap)

# ---------------- DISPLAY ---------------- #
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title("Original MRI")
plt.imshow(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Grad-CAM")
plt.imshow(cv2.cvtColor(gradcam_image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.show()