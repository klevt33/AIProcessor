import logging
import random
from collections import Counter

from classifier_config import ClassifierConfig
from classifier_constants import TrainingDataLabels
from classifier_sql import get_all_classifier_training_data, update_classifier_training_data_version
from training_data import TrainingData

from config import Config
from lemmatizer import get_lemmatizer
from sdp import SDP
from utils import clean_text_for_classification


class DataManager:
    def __init__(self, config: Config, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

        self.sdp = SDP(config=config)

    async def get_training_data(self, classes: list[str] | None = None) -> list[TrainingData]:
        """Retrieve ALL training data that is on of the given classes

        Args:
            classes (list[str] | None): Returned training data must be one of these classes

        Returns:
            list[TrainingData]: The training data that is one of the given classes
        """
        self.logger.info(f"Retrieving training data for classes: {classes}")

        df = await get_all_classifier_training_data(self.sdp, classes)
        return [TrainingData(row) for row in df.to_dict(orient="records")]

    async def update_training_data_version(self, training_data: list[TrainingData]):
        await update_classifier_training_data_version(self.sdp, training_data)

    def balance_training_data(self, training_data: list[TrainingData], classifier_config: ClassifierConfig):
        filtered_data = [d for d in training_data if getattr(d, classifier_config.col_name) in classifier_config.db_classes]
        counts = Counter(getattr(d, classifier_config.col_name) for d in training_data)
        most_common, second_common = counts.most_common(2)
        second_common_count = second_common[1]

        final_training_data = [d for d in filtered_data if getattr(d, classifier_config.col_name) != most_common[0]]
        most_common_filtered = [d for d in filtered_data if getattr(d, classifier_config.col_name) == most_common[0]]
        most_common_new = [
            d for d in most_common_filtered if d.training_version[classifier_config.classifier] == TrainingDataLabels.NEW
        ][:second_common_count]
        final_training_data.extend(most_common_new)
        remaining_data_count = second_common_count - len(most_common_new)
        if remaining_data_count > 0:
            most_common_used = [
                d for d in most_common_filtered if d.training_version[classifier_config.classifier] == TrainingDataLabels.USED
            ]
            final_training_data.extend(random.sample(most_common_used, remaining_data_count))
        return final_training_data

    def clean_data(self, training_data: list[TrainingData]) -> list[TrainingData]:
        lemmatizer = get_lemmatizer()
        for datum in training_data:
            datum.invoice_description = clean_text_for_classification(lemmatizer, datum.invoice_description)
        return training_data

    def remove_duplicates(self, training_data: list[TrainingData]) -> list[TrainingData]:
        seen_descriptions = set()
        return_list = []
        for datum in training_data:
            if datum.invoice_description not in seen_descriptions:
                seen_descriptions.add(datum.invoice_description)
                return_list.append(datum)
        self.logger.info(f"Removed {len(training_data) - len(return_list)} items")
        return return_list
