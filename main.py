from model_evaluation import ModelEvaluation
from model_training import X_test_seq, y_test_seq

evaluator = ModelEvaluation()

metrics = evaluator.evaluate(
    X_test_seq,
    y_test_seq
)

print(metrics)