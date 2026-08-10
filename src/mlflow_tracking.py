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

    def __init__(
        self,
        experiment_name="IBM-Stock-Prediction",
        tracking_uri=None,
    ):
        self.experiment_name = experiment_name
        self.enabled = mlflow is not None
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI",
            "http://127.0.0.1:5000",
        )

        if self.enabled:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)

    def start_run(self):
        if not self.enabled:
            return _NullContext()
        return mlflow.start_run(run_name=self.experiment_name)

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