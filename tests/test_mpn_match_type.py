import unittest

from matching_scores import mpn_match_type


class TestMpnMatchType(unittest.TestCase):
    def test_example_cases(self):
        """Test the specific examples provided in the requirements"""
        description = "3M XYZ ELECTRIC PART PD345123456ABC"

        # Full match case - should return 1
        self.assertEqual(mpn_match_type("PD345123456ABC", description), 1)

        # Partial match cases - should return 0
        self.assertEqual(mpn_match_type("PD345", description), 0)
        self.assertEqual(mpn_match_type("456ABC", description), 0)

        # No match / embedded match case - should return -1
        # "123456" is embedded between "PD345" and "ABC"
        self.assertEqual(mpn_match_type("123456", description), -1)

    def test_separator_removal(self):
        """Test that separators are properly removed"""
        # Test dot separator removal
        self.assertEqual(mpn_match_type("ABC.123", "XYZ ABC123 DEF"), 1)

        # Test dash separator removal
        self.assertEqual(mpn_match_type("ABC-123", "XYZ ABC123 DEF"), 1)

        # Test slash separator removal
        self.assertEqual(mpn_match_type("ABC/123", "XYZ ABC123 DEF"), 1)

        # Space is NOT removed as a separator
        self.assertEqual(mpn_match_type("ABC 123", "XYZ ABC123 DEF"), -1)

    def test_at_beginning(self):
        """Test part number at the beginning of description"""
        # At beginning with space after = match with no chars on either side
        self.assertEqual(mpn_match_type("ABC123", "ABC123 XYZ"), 1)

        # At beginning with lowercase letter after = match with char on one side
        self.assertEqual(mpn_match_type("ABC123", "ABC123XYZ"), 0)

    def test_at_end(self):
        """Test part number at the end of description"""
        # At end with space before = match with no chars on either side
        self.assertEqual(mpn_match_type("ABC123", "XYZ ABC123"), 1)

        # At end with lowercase letter before = match with char on one side
        self.assertEqual(mpn_match_type("ABC123", "XYZABC123"), 0)

    def test_surrounded_by_spaces(self):
        """Test part number surrounded by spaces"""
        self.assertEqual(mpn_match_type("ABC123", "XYZ ^ABC123. DEF"), 1)

    def test_surrounded_by_capital_letters(self):
        """Test part number surrounded by capital letters (should be match)"""
        self.assertEqual(mpn_match_type("ABC123", "XYZABC123DEF"), -1)

    def test_lowercase_boundary(self):
        """Test with lowercase letters at boundaries"""
        # Lowercase before, uppercase after
        self.assertEqual(mpn_match_type("ABC123", "XYZABC123DEF"), -1)

    def test_numeric_boundary(self):
        """Test with numbers at boundaries"""
        # Number before, uppercase after
        self.assertEqual(mpn_match_type("ABC123", "123ABC123DEF"), -1)

        # Uppercase before, number after
        self.assertEqual(mpn_match_type("ABC123", "XYZABC123456"), -1)

        # Number on both sides
        self.assertEqual(mpn_match_type("ABC123", "123ABC123456"), -1)

    def test_no_match(self):
        """Test cases where part number isn't in description"""
        self.assertEqual(mpn_match_type("ABC123", "XYZ DEF456 GHI"), -1)
        self.assertEqual(mpn_match_type("ABC123", ""), -1)

    def test_part_num_is_substring(self):
        """Test where part number is a substring of a larger word"""
        # Embedded in a word with lowercase letters
        self.assertEqual(mpn_match_type("ABC", "XYZABCDEF"), -1)

        # Embedded with uppercase before and number after
        self.assertEqual(mpn_match_type("ABC", "XYZABC123"), -1)

    def test_multiple_occurrences(self):
        """Test with multiple occurrences - should find best match"""
        # Second occurrence has clean boundaries
        self.assertEqual(mpn_match_type("ABC", "XYZAABC DEF (A/B-C) GHI"), 1)

        # First occurrence has one clean boundary, second has both dirty
        self.assertEqual(mpn_match_type("ABC", "XYZ ABC123 XYZABCDEF"), 0)

        # First occurrence has both dirty, second has one clean boundary
        self.assertEqual(mpn_match_type("ABC", "xyz1ABC2 XYZ ABCDEF"), 0)

        # Multiple occurrences but all have both sides dirty
        self.assertEqual(mpn_match_type("ABC", "xyz1ABC2 123ABCDEF"), -1)

    def test_case_sensitivity(self):
        """Test case sensitivity of the match"""
        # Match is case sensitive
        self.assertEqual(mpn_match_type("ABC123", "XYZ ABC123 DEF"), 1)

    def test_exact_match(self):
        """Test when part number is the entire description"""
        self.assertEqual(mpn_match_type("ABC123", "ABC123"), 1)


if __name__ == "__main__":
    unittest.main()
