import os
import sys
from src.exception import CustomException
from src.logger import logging


class DataIngestion:
    def __init__(self):
        self.raw_data_path = "artifacts/data"

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")

        try:
            os.makedirs(self.raw_data_path, exist_ok=True)

            # Placeholder (replace with actual dataset path)
            logging.info("Dataset ingestion completed")

            return self.raw_data_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()