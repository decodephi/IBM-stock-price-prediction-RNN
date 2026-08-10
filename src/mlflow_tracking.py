import os

try:
    import mlflow
except ModuleNotFoundError:
    mlflow = None


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class MLflowTracking:

    def __init__(self, experiment_name="IBM-Stock-Prediction"):
        self.experiment_name = experiment_name
        self.enabled = mlflow is not None

        if self.enabled:
            mlflow.set_experiment(self.experiment_name)

    def start_run(self):
        if not self.enabled:
            return _NullContext()
        return mlflow.start_run()

    def log_parameters(self, params):
        if self.enabled:
            mlflow.log_params(params)

    def log_metrics(self, metrics):
        if self.enabled:
            mlflow.log_metrics(metrics)

    def log_artifact(self, file_path):
        if not self.enabled:
            return

        if os.path.exists(file_path):
            mlflow.log_artifact(file_path)
        else:
            print(f"Artifact not found: {file_path}")

    def end_run(self):
        if self.enabled:
            mlflow.end_run()