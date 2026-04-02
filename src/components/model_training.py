import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.exception import CustomException
from src.logger import logging


class ModelTrainer:
    def __init__(self):
        self.train_dir = "artifacts/train_data"
        self.test_dir = "artifacts/test_data"
        self.model_path = "artifacts/model.h5"

    # ---------------- MODEL ---------------- #
    def build_model(self):
        model = models.Sequential([
            layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
            layers.MaxPooling2D(2,2),

            layers.Conv2D(64, (3,3), activation='relu', name="last_conv"),
            layers.MaxPooling2D(2,2),

            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(2, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    # ---------------- GRAPHS ---------------- #
    def plot_training(self, history):
        plt.figure(figsize=(10,5))

        # Accuracy
        plt.subplot(1,2,1)
        plt.plot(history.history['accuracy'], label='train_acc')
        plt.plot(history.history['val_accuracy'], label='val_acc')
        plt.title("Accuracy")
        plt.legend()

        # Loss
        plt.subplot(1,2,2)
        plt.plot(history.history['loss'], label='train_loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.title("Loss")
        plt.legend()

        os.makedirs("artifacts", exist_ok=True)
        plt.savefig("artifacts/training_plot.png")
        plt.close()

    # ---------------- GRAD CAM ---------------- #
    def generate_gradcam(self, model, image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (128,128))
        input_image = image / 255.0
        input_image = np.expand_dims(input_image, axis=0)

        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer("last_conv").output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(input_image)
            class_index = tf.argmax(predictions[0])
            loss = predictions[:, class_index]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = np.maximum(heatmap, 0) / tf.reduce_max(heatmap)

        heatmap = cv2.resize(heatmap.numpy(), (128,128))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        superimposed = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

        cv2.imwrite("artifacts/gradcam_output.jpg", superimposed)

    # ---------------- TRAINING ---------------- #
    def train_model(self):
        try:
            logging.info("Training started")

            train_gen = ImageDataGenerator(rescale=1./255)
            test_gen = ImageDataGenerator(rescale=1./255)

            train_data = train_gen.flow_from_directory(
                self.train_dir,
                target_size=(128,128),
                batch_size=32,
                class_mode='categorical'
            )

            test_data = test_gen.flow_from_directory(
                self.test_dir,
                target_size=(128,128),
                batch_size=32,
                class_mode='categorical'
            )

            model = self.build_model()

            history = model.fit(
                train_data,
                validation_data=test_data,
                epochs=5
            )

            # Save model
            model.save(self.model_path)

            # Save graphs
            self.plot_training(history)

            # Generate Grad-CAM on 1 sample image
            sample_class = os.listdir(self.train_dir)[0]
            sample_image = os.listdir(os.path.join(self.train_dir, sample_class))[0]
            sample_path = os.path.join(self.train_dir, sample_class, sample_image)

            self.generate_gradcam(model, sample_path)

            logging.info("Training completed successfully")

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = ModelTrainer()
    obj.train_model()