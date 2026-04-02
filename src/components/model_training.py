import os
import sys
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.exception import CustomException
from src.logger import logging


class ModelTrainer:
    def __init__(self):
        self.train_dir = "artifacts/train_data"
        self.test_dir = "artifacts/test_data"
        self.model_path = "artifacts/model.h5"

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

    def train_model(self):
        try:
            logging.info("Starting training...")

            # Data Generators
            train_datagen = ImageDataGenerator(rescale=1./255)
            test_datagen = ImageDataGenerator(rescale=1./255)

            train_generator = train_datagen.flow_from_directory(
                self.train_dir,
                target_size=(128, 128),
                batch_size=32,
                class_mode='categorical'
            )

            test_generator = test_datagen.flow_from_directory(
                self.test_dir,
                target_size=(128, 128),
                batch_size=32,
                class_mode='categorical'
            )

            model = self.build_model()

            model.fit(
                train_generator,
                validation_data=test_generator,
                epochs=5
            )

            # Save model
            os.makedirs("artifacts", exist_ok=True)
            model.save(self.model_path)

            logging.info("Model training completed and saved")

            return model

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_model()