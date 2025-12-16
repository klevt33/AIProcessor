import unittest

# Step 1: Import the actual components from your module
from matching_utils import SpecialCharIgnoringDict, find_manufacturers_in_description

# --- The Test Suite using unittest ---


class TestFindManufacturers(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Set up a class-level manufacturer dictionary. All keys are uppercase,
        which is good practice.
        """
        raw_map = {
            "3M": "3M INC.",
            "3M INC": "3M INC.",
            "AMP": "AMP",
            "AMP INC": "AMP",
            "TYCO": "TYCO ELECTRONICS",
            "TYCO ELECTRONICS": "TYCO ELECTRONICS",
            "MICROCHIP": "MICROCHIP TECHNOLOGY",
            "PROCTER & GAMBLE": "P&G",
        }
        cls.manufacturer_dict = SpecialCharIgnoringDict(raw_map)

    def test_provided_positive_examples(self):
        """Tests the positive matching examples with uppercase input."""
        self.assertEqual(find_manufacturers_in_description(self.manufacturer_dict, "ABC 3M INC ITEM 123"), {"3M INC.": "3M INC"})
        self.assertEqual(
            find_manufacturers_in_description(self.manufacturer_dict, "ABCD 3M INC. ITEM 123"), {"3M INC.": "3M INC"}
        )
        self.assertEqual(
            find_manufacturers_in_description(self.manufacturer_dict, "ABC DEF 3M-INC ITEM 123"), {"3M INC.": "3M INC"}
        )
        self.assertEqual(
            find_manufacturers_in_description(self.manufacturer_dict, "ABCD EFG 3-M (INC.) 123"), {"3M INC.": "3 M INC"}
        )
        self.assertEqual(
            find_manufacturers_in_description(self.manufacturer_dict, "ABCDE 3-MINC. ITEM 123"), {"3M INC.": "3 MINC"}
        )

    def test_longest_match_wins(self):
        """Ensures that when 'AMP' and 'AMP INC' are possible, 'AMP INC' is chosen."""
        description = "ITEM FROM AMP INC FOR TESTING"
        self.assertEqual(find_manufacturers_in_description(self.manufacturer_dict, description), {"AMP": "AMP INC"})

    def test_multiple_manufacturers(self):
        """Tests a description containing multiple different manufacturers."""
        description = "A TYCO PART AND A MICROCHIP PROCESSOR"
        expected = {"TYCO ELECTRONICS": "TYCO", "MICROCHIP TECHNOLOGY": "MICROCHIP"}
        self.assertEqual(find_manufacturers_in_description(self.manufacturer_dict, description), expected)

    def test_multiple_variants_of_same_manufacturer(self):
        """Tests that if 'TYCO' and 'TYCO ELECTRONICS' appear, the longer one is chosen."""
        description = "THIS IS A TYCO CONNECTOR, MADE BY TYCO ELECTRONICS"
        expected = {"TYCO ELECTRONICS": "TYCO ELECTRONICS"}
        self.assertEqual(find_manufacturers_in_description(self.manufacturer_dict, description), expected)

    def test_no_matches(self):
        """Tests a description with no known manufacturers."""
        description = "A GENERIC RESISTOR AND CAPACITOR"
        self.assertEqual(find_manufacturers_in_description(self.manufacturer_dict, description), {})

    def test_empty_inputs(self):
        """Tests behavior with empty description or dictionary."""
        m_dict = SpecialCharIgnoringDict({"TEST": "TEST INC"})
        self.assertEqual(find_manufacturers_in_description(m_dict, ""), {})
        # UPDATED: Use uppercase description for consistency
        self.assertEqual(find_manufacturers_in_description(SpecialCharIgnoringDict(), "A DESCRIPTION"), {})
        self.assertEqual(find_manufacturers_in_description(SpecialCharIgnoringDict(), ""), {})

    def test_complex_characters(self):
        """
        Tests matching with special characters in an uppercase description.
        (Renamed from test_complex_characters_and_case)
        """
        # UPDATED: Description is now fully uppercase.
        description = "A PRODUCT FROM PROCTER & GAMBLE (P&G)"

        # UPDATED: The expected matched text is also uppercase.
        expected = {"P&G": "PROCTER GAMBLE"}
        self.assertEqual(find_manufacturers_in_description(self.manufacturer_dict, description), expected)

    def test_word_boundary_logic(self):
        """Tests how the n-gram logic handles word boundaries."""
        description1 = "ABC 3M INCITEM 123"
        expected1 = {"3M INC.": "3M"}
        self.assertEqual(find_manufacturers_in_description(self.manufacturer_dict, description1), expected1)

        description2 = "ABCD3M INC. ITEM 123"
        self.assertEqual(find_manufacturers_in_description(self.manufacturer_dict, description2), {})

        description3 = "ABCD3MINC1 ITEM 123"
        self.assertEqual(find_manufacturers_in_description(self.manufacturer_dict, description3), {})


# This makes the script runnable from the command line
if __name__ == "__main__":
    unittest.main()
