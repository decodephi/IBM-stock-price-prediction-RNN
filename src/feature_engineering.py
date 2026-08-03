import os
import sys
import yaml

from logger import logger
from exception import CustomException


class FeatureEngineering:

    def __init__(self, df):

        self.df = df

        with open("config/config.yaml", "r") as file:
            config = yaml.safe_load(file)

        self.output_path = config["feature_engineering"]["featured_data_path"]

    def engineer_features(self):

        try:

            logger.info("Starting Feature Engineering...")

            self.df["Year"] = self.df["Date"].dt.year
            self.df["Month"] = self.df["Date"].dt.month
            self.df["Quarter"] = self.df["Date"].dt.quarter
            self.df["Day"] = self.df["Date"].dt.day
            self.df["DayOfWeek"] = self.df["Date"].dt.dayofweek

            logger.info("Calendar features created successfully.")
            ##########################################################
            logger.info("Creating lag features...")

            self.df["Lag_1"] = self.df["Close"].shift(1)
            self.df["Lag_7"] = self.df["Close"].shift(7)
            self.df["Lag_30"] = self.df["Close"].shift(30)

            logger.info("Lag features created successfully.")

            os.makedirs(
                os.path.dirname(self.output_path),
                exist_ok=True
            )

            self.df.to_csv(
                self.output_path,
                index=False
            )

            logger.info(
                f"Feature dataset saved to {self.output_path}"
            )

            return self.df

        except Exception as e:

            logger.error(e)

            raise CustomException(e, sys)