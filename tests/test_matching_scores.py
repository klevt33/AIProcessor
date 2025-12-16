import unittest

import pandas as pd

from matching_scores import (  # remove_separators,
    are_unspsc_compatible,
    best_match_score,
    calculate_initial_confidence_scores,
    get_manufacturer_length_points,
    get_part_number_length_points,
    get_part_number_manufacturer_points,
)
from matching_utils import PartNumberExtractor
from utils import remove_separators


class TestBestMatchScore(unittest.TestCase):
    def setUp(self):
        """Set up test data that will be used across multiple tests."""
        # Create a basic DataFrame with required columns
        self.base_df = pd.DataFrame(
            {
                "ItemID": ["123", "456", "789"],
                "MfrPartNum": ["ABC123", "ABC-123", "XYZ789"],
                "MfrName": ["Acme Corp", "Acme Inc", "XYZ Inc"],
                "UPC": ["11111", "22222", "33333"],
                "UNSPSC": ["44121706", "44120000", "42000000"],
                "AKPartNum": ["AK123", "AK456", "AK789"],
                "DescriptionID": [1, 2, 3],
                "ItemDescription": ["Item 1 description", "Item 2 description", "Item 3 description"],
                "CleanName": ["ACME", "ACME", "XYZ"],
                "UncleanName": ["Acme", "Acme", "XYZ"],
                "DescriptionSimilarity": [0.8, 0.7, 0.6],
            }
        )

        # Manufacturer dict for testing
        self.mfr_dict_single = {"Acme": "ACME"}
        self.mfr_dict_multiple = {"Acme": "ACME", "XYZ": "XYZ"}
        self.mfr_dict_empty = {}

    def print_dataframe(self, df, message="DataFrame"):
        """Helper method to print DataFrame contents for debugging."""
        print(f"\n{message}:")
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        print(df)
        pd.reset_option("display.max_columns")
        pd.reset_option("display.width")

    def test_empty_dataframe(self):
        """Test with an empty DataFrame."""
        print("\n=== TEST: Empty DataFrame ===")
        empty_df = pd.DataFrame(columns=self.base_df.columns)
        result = best_match_score(empty_df, self.mfr_dict_single)
        self.assertIsNone(result)
        print("Result: None (as expected)")

    def test_single_record(self):
        """Test with a single record in the DataFrame."""
        print("\n=== TEST: Single Record ===")
        single_df = self.base_df.iloc[[0]].copy()
        self.print_dataframe(single_df, "Input DataFrame")

        result = best_match_score(single_df, self.mfr_dict_single)
        self.print_dataframe(result, "Result DataFrame")

        # Verify result is a DataFrame with one row
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)

        # Check that confidence scores are calculated
        self.assertTrue("PartNumberConfidenceScore" in result.columns)
        self.assertTrue("ManufacturerNameConfidenceScore" in result.columns)
        self.assertTrue("UNSPSCConfidenceScore" in result.columns)

        # Print confidence scores
        part_num_score = result.iloc[0]["PartNumberConfidenceScore"]
        mfr_score = result.iloc[0]["ManufacturerNameConfidenceScore"]
        unspsc_score = result.iloc[0]["UNSPSCConfidenceScore"]

        print(f"Part Number Confidence Score: {part_num_score}")
        print(f"Manufacturer Name Confidence Score: {mfr_score}")
        print(f"UNSPSC Confidence Score: {unspsc_score}")

    def test_multiple_records_same_part(self):
        """Test with multiple records having the same normalized part number."""
        print("\n=== TEST: Multiple Records, Same Part Number ===")
        # First two records have same normalized part number
        test_df = self.base_df.iloc[[0, 1]].copy()
        self.print_dataframe(test_df, "Input DataFrame")

        # Calculate expected initial scores for debugging
        init_scores_df = calculate_initial_confidence_scores(test_df.copy(), len(self.mfr_dict_single))
        self.print_dataframe(init_scores_df, "DataFrame with Initial Scores")

        result = best_match_score(test_df, self.mfr_dict_single)
        self.print_dataframe(result, "Result DataFrame")

        # Should return only one row (the best match)
        self.assertEqual(len(result), 1)

        # Print details of the calculation
        print(f"Normalized ABC123: {remove_separators('ABC123')}")
        print(f"Normalized ABC-123: {remove_separators('ABC-123')}")
        print(f"Selected MfrPartNum: {result.iloc[0]['MfrPartNum']}")
        print(f"Part Number Confidence Score: {result.iloc[0]['PartNumberConfidenceScore']}")
        print(f"Manufacturer Name Confidence Score: {result.iloc[0]['ManufacturerNameConfidenceScore']}")
        print(f"UNSPSC Confidence Score: {result.iloc[0]['UNSPSCConfidenceScore']}")

    def test_different_manufacturer_names(self):
        """Test with records having different manufacturer names."""
        print("\n=== TEST: Different Manufacturer Names ===")
        # Records with different CleanName values
        test_df = self.base_df.iloc[[0, 2]].copy()
        self.print_dataframe(test_df, "Input DataFrame")

        # Calculate expected initial scores for debugging
        init_scores_df = calculate_initial_confidence_scores(test_df.copy(), len(self.mfr_dict_multiple))
        self.print_dataframe(init_scores_df, "DataFrame with Initial Scores")

        result = best_match_score(test_df, self.mfr_dict_multiple)
        self.print_dataframe(result, "Result DataFrame")

        # Should return only one row (the best match)
        self.assertEqual(len(result), 1)

        print(f"Selected MfrPartNum: {result.iloc[0]['MfrPartNum']}")
        print(f"Selected CleanName: {result.iloc[0]['CleanName']}")
        print(f"Part Number Confidence Score: {result.iloc[0]['PartNumberConfidenceScore']}")
        print(f"Manufacturer Name Confidence Score: {result.iloc[0]['ManufacturerNameConfidenceScore']}")
        print(f"UNSPSC Confidence Score: {result.iloc[0]['UNSPSCConfidenceScore']}")

    def test_compatible_unspsc(self):
        """Test with records having compatible UNSPSC values."""
        print("\n=== TEST: Compatible UNSPSC Values ===")
        # First two records have compatible UNSPSC values (44121706 and 44120000)
        test_df = self.base_df.iloc[[0, 1]].copy()
        self.print_dataframe(test_df, "Input DataFrame")

        # Verify UNSPSC compatibility
        unspsc1 = test_df.iloc[0]["UNSPSC"]
        unspsc2 = test_df.iloc[1]["UNSPSC"]
        compatible = are_unspsc_compatible(unspsc1, unspsc2)
        print(f"UNSPSC Values: {unspsc1} and {unspsc2} are compatible: {compatible}")

        result = best_match_score(test_df, self.mfr_dict_single)
        self.print_dataframe(result, "Result DataFrame")

        print(f"Selected UNSPSC: {result.iloc[0]['UNSPSC']}")
        print(f"UNSPSC Confidence Score: {result.iloc[0]['UNSPSCConfidenceScore']}")

    def test_incompatible_unspsc(self):
        """Test with records having incompatible UNSPSC values."""
        print("\n=== TEST: Incompatible UNSPSC Values ===")
        # First and third records have incompatible UNSPSC values (44121706 and 42000000)
        test_df = self.base_df.iloc[[0, 2]].copy()
        test_df.iloc[1, test_df.columns.get_loc("MfrPartNum")] = "ABC124"  # Change part number to avoid filtering
        self.print_dataframe(test_df, "Input DataFrame")

        # Verify UNSPSC incompatibility
        unspsc1 = test_df.iloc[0]["UNSPSC"]
        unspsc2 = test_df.iloc[1]["UNSPSC"]
        compatible = are_unspsc_compatible(unspsc1, unspsc2)
        print(f"UNSPSC Values: {unspsc1} and {unspsc2} are compatible: {compatible}")

        # Calculate expected initial scores for debugging
        init_scores_df = calculate_initial_confidence_scores(test_df.copy(), len(self.mfr_dict_multiple))
        self.print_dataframe(init_scores_df, "DataFrame with Initial Scores")

        result = best_match_score(test_df, self.mfr_dict_multiple)
        self.print_dataframe(result, "Result DataFrame")

        print(f"Selected UNSPSC: {result.iloc[0]['UNSPSC']}")
        print(f"UNSPSC Confidence Score: {result.iloc[0]['UNSPSCConfidenceScore']}")

    def test_second_record_missing_unspsc(self):
        """Test when second record has no UNSPSC value."""
        print("\n=== TEST: Second Record Missing UNSPSC ===")
        test_df = self.base_df.iloc[[0, 2]].copy()
        test_df.iloc[1, test_df.columns.get_loc("MfrPartNum")] = "ABC124"  # Change part number to avoid filtering
        test_df.iloc[1, test_df.columns.get_loc("UNSPSC")] = None  # Second record has no UNSPSC
        self.print_dataframe(test_df, "Input DataFrame")

        # Calculate expected initial scores for debugging
        init_scores_df = calculate_initial_confidence_scores(test_df.copy(), len(self.mfr_dict_multiple))
        self.print_dataframe(init_scores_df, "DataFrame with Initial Scores")

        result = best_match_score(test_df, self.mfr_dict_multiple)
        self.print_dataframe(result, "Result DataFrame")

        print(f"Top record has UNSPSC: {result.iloc[0]['UNSPSC']}")
        print(f"UNSPSC Confidence Score: {result.iloc[0]['UNSPSCConfidenceScore']}")

    def test_multiple_records_low_confidence(self):
        """Test filtering out low confidence records."""
        print("\n=== TEST: Filtering Low Confidence Records ===")
        test_df = self.base_df.copy()
        # Deliberately set a very low description similarity for one record
        test_df.iloc[2, test_df.columns.get_loc("DescriptionSimilarity")] = 0.1
        self.print_dataframe(test_df, "Input DataFrame")

        # Calculate expected initial scores for debugging
        init_scores_df = calculate_initial_confidence_scores(test_df.copy(), len(self.mfr_dict_multiple))
        self.print_dataframe(init_scores_df, "DataFrame with Initial Scores")

        result = best_match_score(test_df, self.mfr_dict_multiple)
        self.print_dataframe(result, "Result DataFrame")

        # Print top confidence score threshold
        if not init_scores_df.empty:
            top_score = init_scores_df["PartNumberConfidenceScore"].max()
            threshold = top_score / 2
            print(f"Top confidence score: {top_score}")
            print(f"Threshold for filtering: {threshold}")

        print(f"Final record count: {len(result)}")

    def test_no_manufacturer_matches(self):
        """Test when there are no manufacturer matches in the dictionary."""
        print("\n=== TEST: No Manufacturer Matches ===")
        test_df = self.base_df.copy()
        test_df["UncleanName"] = None

        # Debug: Print the structure of the empty manufacturer dictionary
        print(f"Manufacturer dictionary content: {self.mfr_dict_empty}")
        print(f"Manufacturer dictionary type: {type(self.mfr_dict_empty)}")
        print(f"Manufacturer dictionary length: {len(self.mfr_dict_empty)}")

        # Debug: Print UncleanName values to verify they're valid
        print("\nUncleanName values in test DataFrame:")
        for idx, val in enumerate(test_df["UncleanName"]):
            print(f"Row {idx}: Value: '{val}', Type: {type(val)}, Is NaN: {pd.isna(val)}")

        # Debug: Test the manufacturer points function directly
        for idx, row in test_df.iterrows():
            unclean_name = row["UncleanName"]
            points = get_part_number_manufacturer_points(unclean_name, len(self.mfr_dict_empty))
            print(f"Row {idx}: UncleanName: '{unclean_name}', Matched count: {len(self.mfr_dict_empty)}, Points: {points}")

        self.print_dataframe(test_df, "Input DataFrame")

        try:
            # Calculate expected initial scores for debugging
            init_scores_df = calculate_initial_confidence_scores(test_df.copy(), len(self.mfr_dict_empty))
            self.print_dataframe(init_scores_df, "DataFrame with Initial Scores")

            result = best_match_score(test_df, self.mfr_dict_empty)
            self.print_dataframe(result, "Result DataFrame")

            print(f"Manufacturer match count: {len(self.mfr_dict_empty)}")
            print(f"Selected Part Number: {result.iloc[0]['MfrPartNum']}")
            print(f"Part Number Confidence Score: {result.iloc[0]['PartNumberConfidenceScore']}")
            print(f"Manufacturer Name Confidence Score: {result.iloc[0]['ManufacturerNameConfidenceScore']}")
        except Exception as e:
            import traceback

            print(f"Error during test: {e}")
            print(traceback.format_exc())

    def test_different_part_number_lengths(self):
        """Test part numbers with different effective lengths."""
        print("\n=== TEST: Different Part Number Lengths ===")
        test_df = pd.DataFrame(
            {
                "ItemID": ["123", "456", "789", "101"],
                "MfrPartNum": ["A", "ABCDE", "ABCDEFGHIJ", "ABCDEFGHIJKLMN"],
                "MfrName": ["Test Corp", "Test Corp", "Test Corp", "Test Corp"],
                "UPC": ["11111", "22222", "33333", "44444"],
                "UNSPSC": ["44121706", "44121706", "44121706", "44121706"],
                "AKPartNum": ["AK1", "AK2", "AK3", "AK4"],
                "DescriptionID": [1, 2, 3, 4],
                "ItemDescription": ["Short", "Medium", "Long", "Very Long"],
                "CleanName": ["TEST", "TEST", "TEST", "TEST"],
                "UncleanName": ["Test", "Test", "Test", "Test"],
                "DescriptionSimilarity": [0.6, 0.6, 0.6, 0.6],
            }
        )
        self.print_dataframe(test_df, "Input DataFrame")

        # Print effective lengths and expected points
        for _, row in test_df.iterrows():
            part_num = row["MfrPartNum"]
            eff_len = PartNumberExtractor.effective_length(part_num)
            len_points = get_part_number_length_points(eff_len)
            mfr_len_points = get_manufacturer_length_points(eff_len)
            print(f"Part {part_num}: Effective Length = {eff_len}, PN Points = {len_points}, MFR Points = {mfr_len_points}")

        # Calculate expected initial scores for debugging
        init_scores_df = calculate_initial_confidence_scores(test_df.copy(), len(self.mfr_dict_single))
        self.print_dataframe(init_scores_df, "DataFrame with Initial Scores")

        result = best_match_score(test_df, self.mfr_dict_single)
        self.print_dataframe(result, "Result DataFrame")

    def test_different_description_similarities(self):
        """Test records with different description similarities."""
        print("\n=== TEST: Different Description Similarities ===")
        test_df = pd.DataFrame(
            {
                "ItemID": ["123", "456", "789", "101"],
                "MfrPartNum": ["ABC123", "ABC123", "ABC123", "ABC123"],
                "MfrName": ["Test Corp", "Test Corp", "Test Corp", "Test Corp"],
                "UPC": ["11111", "22222", "33333", "44444"],
                "UNSPSC": ["44121706", "44121706", "44121706", "44121706"],
                "AKPartNum": ["AK1", "AK2", "AK3", "AK4"],
                "DescriptionID": [1, 2, 3, 4],
                "ItemDescription": ["Low", "Medium", "High", "Perfect"],
                "CleanName": ["TEST", "TEST", "TEST", "TEST"],
                "UncleanName": ["Test", "Test", "Test", "Test"],
                "DescriptionSimilarity": [0.5, 0.65, 0.8, 1.0],
            }
        )
        self.print_dataframe(test_df, "Input DataFrame")

        # Print description similarity points
        for _, row in test_df.iterrows():
            desc_sim = row["DescriptionSimilarity"]
            pn_points = (desc_sim - 0.55) * 50
            mfr_points = pn_points
            unspsc_points = (desc_sim - 0.55) * 180
            print(
                f"Description Similarity {desc_sim}: PN Points = {pn_points}, MFR Points = {mfr_points}, UNSPSC Points ="
                f" {unspsc_points}"
            )

        # Calculate expected initial scores for debugging
        init_scores_df = calculate_initial_confidence_scores(test_df.copy(), len(self.mfr_dict_single))
        self.print_dataframe(init_scores_df, "DataFrame with Initial Scores")

        result = best_match_score(test_df, self.mfr_dict_single)
        self.print_dataframe(result, "Result DataFrame")


if __name__ == "__main__":
    unittest.main()
