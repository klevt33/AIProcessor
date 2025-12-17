import json
from dataclasses import dataclass

from classifier_constants import ALL_CLASSIFIERS, TrainingDataLabels


@dataclass
class TrainingData:
    """Training data from SDP"""

    invoice_desc_uid: int
    invoice_description: str
    classification: str
    training_version: str
    rental_indicator: str

    def __init__(self, dictionary: dict):
        """Training data from SDP

        Args:
            dictionary (dict): Row-wise dictionary returned from SDP training data table
        """
        self.invoice_desc_uid = dictionary["IVCE_XCTN_CLSFR_TRNL_DESC_REF_UID"]
        self.invoice_description = dictionary["IVCE_PRDT_LDSC"]
        self.classification = dictionary["CLS_NM"]
        self.rental_indicator = dictionary["RNTL_IND"]
        try:
            self.training_version = json.loads(dictionary["TRNG_DAT_VRSN_NUM"])
        except Exception:
            self.training_version = {x: TrainingDataLabels.NEW.value for x in ALL_CLASSIFIERS}

    def get_blob_location(self, training_folder: str) -> str:
        """Return the location to store this training data blob in storage bucket

        Returns:
            str: Location to store this training data blob in storage bucket
        """
        # TODO: make training folder depend on date
        return f"{training_folder}/{self.classification.upper()}/{self._get_blob_id()}"

    def _get_blob_id(self) -> str:
        """Returns the blob id str to be used for uploading to storage container

        Returns:
            str: Blob id str to be used for uploading to storage container
        """
        return f"{self.invoice_desc_uid}.txt"
