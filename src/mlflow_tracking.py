import mlflow
import os


class MLflowTracking:

    def __init__(self, experiment_name="IBM-Stock-Prediction"):

        self.experiment_name = experiment_name

        # Create experiment if it doesn't exist
        mlflow.set_experiment(self.experiment_name)

    def start_run(self):

        return mlflow.start_run()

    def log_parameters(self, params):

        mlflow.log_params(params)

    def log_metrics(self, metrics):

        mlflow.log_metrics(metrics)

    def log_artifact(self, file_path):

        if os.path.exists(file_path):
            mlflow.log_artifact(file_path)
        else:
            print(f"Artifact not found: {file_path}")

    def end_run(self):

        mlflow.end_run()