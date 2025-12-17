import re
from enum import StrEnum
from functools import partial
from typing import Callable

from ai_utils import get_clean_mfr_name
from matching_utils import mfr_eq_type, read_manufacturer_data


class MatchStatus(StrEnum):
    YES = "yes"
    NO = "no"
    SKIP = "skip"


class Matcher:
    def __init__(self, config) -> None:
        self._result_match_methods: dict[str, Callable[[str, str], str]] = {
            "MFR_NAME": self._check_mfr_name_match,
            "MFR_PN": self._check_mfr_pn_name_match,
            "UNSPSC_CODE": self._check_unspsc_match,
        }

        from sdp import SDP

        self.sdp = SDP(config=config)

        self.check_unspsc_2 = partial(self._check_unspsc_match, digits=2)
        self.check_unspsc_4 = partial(self._check_unspsc_match, digits=4)
        self.check_unspsc_6 = partial(self._check_unspsc_match, digits=6)
        self.check_unspsc_8 = partial(self._check_unspsc_match, digits=8)

    async def is_match(self, llm_prected_value, expected_value, match_type) -> str:
        return (await self._result_match_methods[match_type](str(expected_value), str(llm_prected_value))).value

    @staticmethod
    def _clean_string(input_str: str):
        """Remove special characters

        Args:
            input_str (str)

        Returns:
            str: Cleaned input
        """
        return re.sub(r"[^A-Za-z0-9 ]", "", str(input_str).strip())

    @staticmethod
    def _prefix_or_suffix_match(expected_value: str, predicted_value: str) -> bool:
        """Returns True if exp = pred or if one starts with the other or ends with the other.

        Args:
            expected_value (str):
            predicted_value (str):

        Returns:
            bool: True if match, False if no match
        """
        return any(
            (
                expected_value == predicted_value,
                expected_value.startswith(predicted_value),
                expected_value.endswith(predicted_value),
                predicted_value.startswith(expected_value),
                predicted_value.endswith(expected_value),
            )
        )

    async def _check_mfr_name_match(self, expected_value: str, predicted_value: str) -> MatchStatus:
        """Checking if manufacturing name prediction and label match.
        Exact match check first.
        Clean up and prefix/suffix check second.
        Parent/child relationship check third.
        If none of those come back as matches then it is a no match.

        Args:
            expected_value (str):
            predicted_value (str):

        Returns:
            str: "yes" if match, "no" if no match, "skip" if not using for evaluation
        """
        if expected_value == predicted_value:
            return MatchStatus.YES
        if expected_value == "":
            return MatchStatus.SKIP  # not using for evaluation

        expected_value = (await get_clean_mfr_name(self.sdp, expected_value))[0]
        predicted_value = (await get_clean_mfr_name(self.sdp, predicted_value))[0]

        match = self._prefix_or_suffix_match(expected_value, predicted_value)
        if match:
            return MatchStatus.YES

        is_related = bool(mfr_eq_type(expected_value, predicted_value, await read_manufacturer_data(self.sdp)))

        return MatchStatus.YES if is_related else MatchStatus.NO

    @staticmethod
    async def _check_mfr_pn_name_match(expected_value: str, predicted_value: str) -> MatchStatus:
        """Checking if part number prediction and label match.
        It will clean the string first and then does a prefix/suffix check

        Args:
            expected_value (str):
            predicted_value (str):

        Returns:
            str: "yes" if match, "no" if no match, "skip" if not using for evaluation
        """
        expected_value = Matcher._clean_string(expected_value)
        predicted_value = Matcher._clean_string(predicted_value)

        if expected_value == "":
            return MatchStatus.SKIP

        match = Matcher._prefix_or_suffix_match(expected_value, predicted_value)

        return MatchStatus.YES if match else MatchStatus.NO

    @staticmethod
    async def _check_unspsc_match(expected_value: str, predicted_value: str, digits: int = 6) -> MatchStatus:
        """Checks to see if the UNSPSC codes match. The first n (6) digits must match to be successful

        Args:
            expected_value (str):
            predicted_value (str | int):
            digits (int): The number of digits at the beginning that must match to be considered a full match.

        Returns:
            str: "yes" if match, "no" if no match, "skip" if not using for evaluation
        """
        expected_value = expected_value.strip()
        if expected_value in ["-", "", "nan", "R", "LOT"]:
            return MatchStatus.SKIP
        return MatchStatus.YES if expected_value[:digits] == predicted_value[:digits] else MatchStatus.NO
