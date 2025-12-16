from typing import Any

from classifier_config import ClassifierConfig


class Evaluator:
    def __init__(self, classifier_conf: ClassifierConfig) -> None:
        self.classifier_conf = classifier_conf

    @staticmethod
    def _get_macro_f1(evaluation: dict[str, Any]) -> float:
        return evaluation["customSingleLabelClassificationEvaluation"]["macroF1"]

    def meets_threshold(self, evaluation: dict[str, Any]) -> bool:
        return self._get_macro_f1(evaluation) >= self.classifier_conf.threshold

    def beats_current_model(self, new_evaluation: dict[str, Any], current_evaluation: dict[str, Any]) -> bool:
        return self._get_macro_f1(new_evaluation) >= self._get_macro_f1(current_evaluation)
