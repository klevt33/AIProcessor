import unittest

from matching_utils import SpecialCharIgnoringDict, find_manufacturers_in_description


class TestFindManufacturersInDescription(unittest.TestCase):
    def setUp(self):
        raw_dict = {
            "3M": "3M",
            "XYZ Corp": "XYZ Corp",
            "ABC Industries": "ABC Industries",
            "DEF Tech": "DEF Tech",
            "XYZ": "XYZ Corp",  # Alias for XYZ Corp
            "XYZ Corp New": "XYZ New Corporation",
        }

        self.manufacturer_dict = SpecialCharIgnoringDict(raw_dict)

    def test_single_match(self):
        description = "High quality tape from 3M available now."
        expected = {"3M": "3M"}
        self.assertEqual(find_manufacturers_in_description(self.manufacturer_dict, description), expected)

    def test_multiple_matches(self):
        description = "Products from 3M and XYZ Corp are on sale."
        expected = {"3M": "3M", "XYZ Corp": "XYZ Corp"}
        self.assertEqual(find_manufacturers_in_description(self.manufacturer_dict, description), expected)

    def test_alias_match(self):
        description = "XYZ manufactures various products."
        expected = {"XYZ Corp": "XYZ"}
        self.assertEqual(find_manufacturers_in_description(self.manufacturer_dict, description), expected)

    def test_no_match(self):
        description = "This description does not contain any known manufacturer."
        expected = {}
        self.assertEqual(find_manufacturers_in_description(self.manufacturer_dict, description), expected)

    def test_substring_exclusion(self):
        description = "XYZ Corp and XYZ are competitors."
        expected = {"XYZ Corp": "XYZ Corp"}  # "XYZ" should be excluded as a substring
        self.assertEqual(find_manufacturers_in_description(self.manufacturer_dict, description), expected)

    def test_longest_uncleanname_per_cleanname(self):
        description = "DEF Tech and DEF are leading brands."
        expected = {"DEF Tech": "DEF Tech"}  # "DEF" should be excluded
        self.assertEqual(find_manufacturers_in_description(self.manufacturer_dict, description), expected)

    def test_longer_over_shorter(self):
        description = "ABC123 MN: XYZ Corp New"
        expected = {"XYZ New Corporation": "XYZ Corp New"}  # "XYZ" should be excluded
        self.assertEqual(find_manufacturers_in_description(self.manufacturer_dict, description), expected)


if __name__ == "__main__":
    unittest.main()
