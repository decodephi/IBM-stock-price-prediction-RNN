from src.data_ingestion import DataIngestion
from src.data_validation import DataValidation
from src.data_preprocessing import DataPreprocessing
from src.feature_engineering import FeatureEngineering
from src.model_training import ModelTraining
from src.model_evaluation import ModelEvaluation

from src.mlflow_tracking import MLflowTracking


def main():
    
    tracker = MLflowTracking()
    
    with tracker.start_run():
    
    
        ingestion = DataIngestion()
        df = ingestion.load_data()

        # Data Validation
        validator = DataValidation(df)
        validator.validate()

        # Data Preprocessing
        preprocessor = DataPreprocessing(df)
        df = preprocessor.preprocess()

        # Engineer_features
        engineer_features = FeatureEngineering(df)
        df = engineer_features.engineer_features()


        # Model Training
        trainer = ModelTraining()

        X, y = trainer.split_features_target(df)

        X_train, X_test, y_train, y_test = \
        trainer.train_test_split_data(X, y)

        X_train_scaled, X_test_scaled = \
        trainer.scale_data(X_train, X_test)

        X_train_seq, y_train_seq = \
        trainer.create_sequences(X_train_scaled, y_train)

        X_test_seq, y_test_seq = \
        trainer.create_sequences(X_test_scaled, y_test)

        model, history = trainer.train_lstm(
            X_train_seq,
            y_train_seq,
            X_test_seq,
            y_test_seq
        )

        trainer.save_model(model)

        # 6. Model Evaluation
        evaluator = ModelEvaluation()

        metrics = evaluator.evaluate(
            X_test_seq,
            y_test_seq
        )

        print(metrics)


if __name__ == "__main__":
    main()