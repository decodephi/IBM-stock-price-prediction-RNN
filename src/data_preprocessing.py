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


class DataPreprocessing:
    """
    Responsible for preprocessing the dataset.
    Step 3.1: Convert Date column to datetime.
    """

    def __init__(self, df):
        self.df = df

    def preprocess(self):
        """
        Convert Date column to datetime format.
        """

        try:

            logger.info("Starting Data Preprocessing...")

            logger.info("Converting 'Date' column to datetime.")

            self.df["Date"] = pd.to_datetime(self.df["Date"])

            logger.info("Date column converted successfully.")

            logger.info("Data Preprocessing Step 3.1 Completed.")

            return self.df

        except Exception as e:

            logger.error(e)

            raise CustomException(e, sys)