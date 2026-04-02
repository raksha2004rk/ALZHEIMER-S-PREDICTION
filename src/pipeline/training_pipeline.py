from src.components.model_training import ModelTrainer
from src.logger import logging


class TrainingPipeline:
    def start(self):
        logging.info("Pipeline started")

        trainer = ModelTrainer()
        trainer.train_model()

        logging.info("Pipeline completed")


if __name__ == "__main__":
    obj = TrainingPipeline()
    obj.start()