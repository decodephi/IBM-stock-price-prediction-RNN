"""

✅ Convert data types
✅ Handle missing values
✅ Remove duplicate rows
✅ Sort data by date
✅ Reset index
✅ Save processed data
"""


import sys
import pandas as pd

from logger import logger
from exception import CustomException

import yaml
import os


class DataPreprocessing:
    """
    Responsible for preprocessing the dataset.
   
    """

    def __init__(self, df):
     
        with open("config/config.yaml", "r") as file:
            config = yaml.safe_load(file)

        self.processed_data_path = config["data_preprocessing"]["processed_data_path"]

    def preprocess(self):
        """
        Convert Date column to datetime format.
        """

        try:

            logger.info("Starting Data Preprocessing...")
            logger.info("Converting 'Date' column to datetime.")

            self.df["Date"] = pd.to_datetime(self.df["Date"], errors="raise")
            
            # Remove duplicate rows
            duplicate_count = self.df.duplicated().sum()
            logger.info(f"Duplicate rows before removal: {duplicate_count}")
            self.df.drop_duplicates(inplace=True)
            logger.info("Duplicate rows removed successfully.")

            # Handle missing values
            missing_before = self.df.isnull().sum().sum()
            logger.info(f"Total missing values before handling: {missing_before}")
            self.df.dropna(inplace=True)
            missing_after = self.df.isnull().sum().sum()
            logger.info(f"Total missing values after handling: {missing_after}")

            logger.info("Missing values handled successfully.")
            
            
            # Create processed directory if it doesn't exist
            os.makedirs(os.path.dirname(self.processed_data_path), exist_ok=True)
            # Save processed dataset
            self.df.to_csv(self.processed_data_path, index=False)

            logger.info(
                f"Processed dataset saved to: {self.processed_data_path}"
            )
            

            return self.df

        except Exception as e:

            logger.error(e)
            raise CustomException(e, sys)