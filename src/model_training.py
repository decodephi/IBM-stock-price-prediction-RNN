import sys
import yaml
import pandas as pd

from src.logger import logger
from src.exception import CustomException

import os
import joblib

from sklearn.preprocessing import MinMaxScaler

import numpy as np


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

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
        
        self.epochs = config["model_training"]["epochs"]
        self.batch_size = config["model_training"]["batch_size"]
        self.lstm_model_path = config["models"]["lstm_model_path"]
        

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
        
    def train_lstm(self, X_train, y_train, X_test, y_test):
        """
        Train an LSTM model.
        """

        try:
            logger.info("Building LSTM model...")
            model = Sequential()

            model.add(
                LSTM(
                    units=64,
                    return_sequences=False,
                    input_shape=(X_train.shape[1], X_train.shape[2])
                )
            )

            model.add(Dropout(0.2))
            model.add(Dense(32, activation="relu"))
            model.add(Dense(1))

            model.compile(
                optimizer="adam",
                loss="mse",
                metrics=["mae"]
            )

            logger.info("LSTM model built successfully.")

            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True
            )

            logger.info("Training LSTM model...")

            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_test, y_test),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=[early_stop],
                verbose=1
            )

            logger.info("Model training completed.")

            return model, history

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)

    def save_model(self, model):
        """
        Save the trained model.
        """

        try:
            logger.info("Saving trained model...")

            os.makedirs(
                os.path.dirname(self.lstm_model_path),
                exist_ok=True
            )

            model.save(self.lstm_model_path)

            logger.info(
                f"LSTM model saved at {self.lstm_model_path}"
            )

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
    
    model, history = trainer.train_lstm(
    X_train_seq,
    y_train_seq,
    X_test_seq,
    y_test_seq
    )

# Save Model
    trainer.save_model(model)

    print(df.head())
    
    print(X_train_seq.shape)
    print(y_train_seq.shape)