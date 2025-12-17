from datetime import date
from typing import Iterable

from classifier_constants import API_VERSION
from training_data import TrainingData


def get_current_date_str() -> str:
    """Returns the current date as a string ("2025-08-28")

    Returns:
        str: The current date as a string ("2025-08-28")
    """
    return date.today().strftime("%Y-%m-%d")


def training_data_to_labels_file(
    classifier: str,
    container_name: str,
    project_name: str,
    training_data: Iterable[TrainingData],
    classes: list[str],
    training_folder: str,
) -> dict:
    """Generate the labels file from the given training data

    Args:
        classifier (str): name of the classifier
        container_name (str): name of the storage container where the training data is being stored
        project_name (str): name of the language studio project
        training_data (list[TrainingData]): a list of the training data with uid, and classification
        classes (list[str]): a list of all of the possible classes this classifier with classify documents as

    Returns:
        dict: dictionary for the equivalent json labels file
    """
    return {
        "projectFileVersion": API_VERSION,
        "stringIndexType": "Utf16CodeUnit",
        "metadata": {
            "projectName": project_name,
            "storageInputContainerName": container_name,
            "projectKind": "CustomSingleLabelClassification",
            "description": f"Text {classifier} classifier",
            "language": "en-us",
            "multilingual": False,
            "settings": {},
        },
        "assets": {
            "projectKind": "CustomSingleLabelClassification",
            "classes": [{"category": classification} for classification in classes],
            "documents": [
                {
                    "location": training_datum.get_blob_location(training_folder),
                    "language": "en-us",
                    "class": {"category": training_datum.classification},
                }
                for training_datum in training_data
            ],
        },
    }
