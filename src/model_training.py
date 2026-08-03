import sys
import yaml
import pandas as pd

from logger import logger
from exception import CustomException

import os
import joblib

from sklearn.preprocessing import MinMaxScaler

import numpy as np

class ModelTraining:
    """
    Responsible for loading the feature engineered dataset.
    """

    def __init__(self, config_path="config/config.yaml"):

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        # The feature-engineered CSV is defined under the
        # `feature_engineering` section in config.yaml.
        self.feature_data_path = config["feature_engineering"].get(
            "featured_data_path",
            config["model_training"].get("feature_data_path")
        )
        self.scaler_path = config["artifacts"]["scaler_path"]
        self.window_size = config["model_training"]["window_size"]
        

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
        
    def scale_data(self, X_train, X_test):
        """
        Scale training and testing data using MinMaxScaler.
        """

        try:

            logger.info("Starting feature scaling...")

            scaler = MinMaxScaler()

            # Fit only on training data
            X_train_scaled = scaler.fit_transform(X_train)

            # Transform test data
            X_test_scaled = scaler.transform(X_test)

            # Create artifacts directory
            os.makedirs(
            os.path.dirname(self.scaler_path),
            exist_ok=True
        )

           # Save scaler
            joblib.dump(
                scaler,
                self.scaler_path
            )

            logger.info(
                f"Scaler saved at {self.scaler_path}"
            )

            logger.info("Feature scaling completed successfully.")

            return X_train_scaled, X_test_scaled

        except Exception as e:

            logger.error(e)
            raise CustomException(e, sys)
        
        
    def create_sequences(self, X, y):
        """
        Create sliding window sequences for RNN/LSTM/GRU models.
        """

        try:

            logger.info("Creating sliding window sequences...")

            X_seq = []
            y_seq = []

            for i in range(self.window_size, len(X)):

                X_seq.append(
                    X[i - self.window_size:i]
                )

                y_seq.append(
                    y.iloc[i]
                )

            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq)

            logger.info("Sliding window created successfully.")

            logger.info(f"X Shape : {X_seq.shape}")
            logger.info(f"y Shape : {y_seq.shape}")

            return X_seq, y_seq

        except Exception as e:

            logger.error(e)
            raise CustomException(e, sys)



if __name__ == "__main__":

    trainer = ModelTraining()

    df = trainer.load_feature_data()
    
    X, y = trainer.split_features_target(df)


    X_train, X_test, y_train, y_test = trainer.train_test_split_data(
        X,y
    )
    
    X_train_scaled, X_test_scaled = trainer.scale_data(
    X_train,X_test
    )
    
    
    X_train_seq, y_train_seq = trainer.create_sequences(
        X_train_scaled,
        y_train
    )

    X_test_seq, y_test_seq = trainer.create_sequences(
        X_test_scaled,
        y_test
    )

    print(df.head())
    
    print(X_train_seq.shape)
    print(y_train_seq.shape)