import sys
import yaml
import pandas as pd

from logger import logger
from exception import CustomException


class ModelTraining:
    """
    Responsible for loading the feature engineered dataset.
    """

    def __init__(self, config_path="config/config.yaml"):

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        self.feature_data_path = config["model_training"]["feature_data_path"]

    def load_feature_data(self):
        """
        Load the feature engineered dataset.
        """

        try:

            logger.info("Loading feature engineered dataset...")

            df = pd.read_csv(self.feature_data_path)

            logger.info("Feature dataset loaded successfully.")

            logger.info(f"Dataset Shape : {df.shape}")
            
            

            return df

        except Exception as e:

            logger.error(e)

            raise CustomException(e, sys)
        
        

    def split_features_target(self, df):
        """
        Separate features (X) and target (y).
        """

        try:

            logger.info("Separating features and target...")

            # Target variable
            y = df["Close"]

            # Features (remove target and Date)
            X = df.drop(columns=["Date", "Close"])

            logger.info("Feature and target separation completed.")

            logger.info(f"Feature Shape : {X.shape}")
            logger.info(f"Target Shape : {y.shape}")

            return X, y

        except Exception as e:

            logger.error(e)
            raise CustomException(e, sys)
        
    def train_test_split_data(self, X, y, train_size=0.8):
        """
        Split the dataset chronologically into training and testing sets.
        """

        try:

            logger.info("Performing chronological train-test split...")

            split_index = int(len(X) * train_size)

            X_train = X.iloc[:split_index]
            X_test = X.iloc[split_index:]

            y_train = y.iloc[:split_index]
            y_test = y.iloc[split_index:]

            logger.info(f"Training samples: {len(X_train)}")
            logger.info(f"Testing samples: {len(X_test)}")

            return X_train, X_test, y_train, y_test

        except Exception as e:

            logger.error(e)

            raise CustomException(e, sys)



if __name__ == "__main__":

    trainer = ModelTraining()

    df = trainer.load_feature_data()

    print(df.head())