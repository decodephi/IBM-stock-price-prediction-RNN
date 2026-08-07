import os
import sys
import json
import yaml
import numpy as np

from tensorflow.keras.models import load_model

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from logger import logger
from exception import CustomException

class ModelEvaluation:
    """
    Responsible for evaluating the trained model.
    """

    def __init__(self, config_path="config/config.yaml"):

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        self.model_path = config["model_evaluation"]["model_path"]
        self.evaluation_path = config["model_evaluation"]["evaluation_path"]

    def evaluate(self, X_test, y_test):

        try:

            logger.info("Loading trained model...")

            model = load_model(self.model_path)

            logger.info("Model loaded successfully.")

            logger.info("Making predictions...")

            y_pred = model.predict(X_test)

            y_pred = y_pred.flatten()

            logger.info("Predictions completed.")

            # -----------------------
            # Evaluation Metrics
            # -----------------------

            mae = mean_absolute_error(y_test, y_pred)

            mse = mean_squared_error(y_test, y_pred)

            rmse = np.sqrt(mse)

            r2 = r2_score(y_test, y_pred)

            mape = np.mean(
                np.abs((y_test - y_pred) / y_test)
            ) * 100

            metrics = {

                "MAE": float(mae),

                "MSE": float(mse),

                "RMSE": float(rmse),

                "R2": float(r2),

                "MAPE": float(mape)

            }

            logger.info("Evaluation metrics calculated successfully.")

            # -----------------------
            # Save Evaluation Report
            # -----------------------

            os.makedirs(
                os.path.dirname(self.evaluation_path),
                exist_ok=True
            )

            with open(self.evaluation_path, "w") as file:

                json.dump(
                    metrics,
                    file,
                    indent=4
                )

            logger.info(
                f"Evaluation report saved to {self.evaluation_path}"
            )

            return metrics

        except Exception as e:

            logger.error(e)

            raise CustomException(e, sys)