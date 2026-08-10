'''
Note that - Verify that the dataset is valid before sending it to preprocessing.

'''


import sys

from src.logger import logger
from src.exception import CustomException


class DataValidation:
    """
    Responsible for validating the dataset before preprocessing.
    """

    REQUIRED_COLUMNS = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    ]

    def __init__(self, df):
        self.df = df

    def validate(self):
        """
        Validate the input dataset.
        """

        try:

            logger.info("Starting data validation...")

            # 1. Check whether dataset is empty
            if self.df.empty:
                raise ValueError("Dataset is empty.")

            logger.info("Dataset is not empty.")

            # 2. Check required columns
            missing_columns = list(
                set(self.REQUIRED_COLUMNS) - set(self.df.columns)
            )

            if missing_columns:
                raise ValueError(
                    f"Missing columns: {missing_columns}"
                )

            logger.info("All required columns are present.")

            # 3. Count duplicate rows
            duplicate_count = self.df.duplicated().sum()

            logger.info(
                f"Duplicate rows found: {duplicate_count}"
            )

            # 4. Count missing values
            missing_values = self.df.isnull().sum()

            logger.info("Missing values per column:")

            for column, count in missing_values.items():
                logger.info(f"{column}: {count}")

            logger.info("Data Validation Completed Successfully.")

            return True

        except Exception as e:

            logger.error(e)

            raise CustomException(e, sys)
