import unittest

from matching_utils import PartNumberExtractor


class TestPartNumberExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = PartNumberExtractor()

    def test_empty_string(self):
        """Test with empty string"""
        self.assertEqual(self.extractor.effective_length(""), 0)

    def test_basic_length(self):
        """Test basic string length without special rules"""
        # "abc123" has length 6, no capital letters, no repetitions
        self.assertEqual(self.extractor.effective_length("abc123"), 6)

    def test_dash_and_slash_deduction(self):
        """Test deduction of dashes and slashes"""
        # "abc-123" has length 7, minus 1 for dash = 6
        self.assertEqual(self.extractor.effective_length("abc-123"), 6)
        # "abc/123" has length 7, minus 1 for slash = 6
        self.assertEqual(self.extractor.effective_length("abc/123"), 6)
        # "a-b-c/1-2-3" has length 11, minus 5 for dashes/slashes = 6
        self.assertEqual(self.extractor.effective_length("a-b-c/1-2-3"), 6)

    def test_repetition_deduction(self):
        """Test deduction for repeated characters"""
        # "abc000123" has length 9, minus 1 for "000" = 8
        self.assertEqual(self.extractor.effective_length("abc000123"), 8)

        # "aaaa123" has length 7, minus 2 for "aaaa" = 5
        self.assertEqual(self.extractor.effective_length("aaaa123"), 5)

        # "aa000bbb123" has length 11, minus 1 for "000", minus 1 for "bbb" = 9
        self.assertEqual(self.extractor.effective_length("aa000bbb123"), 9)

        # "aa11bb" has length 6, no deductions for repetitions = 6
        self.assertEqual(self.extractor.effective_length("aa11bb"), 6)

    def test_capital_letters_bonus(self):
        """Test bonus for capital letters"""
        # "a" has length 1, no bonus = 1
        self.assertEqual(self.extractor.effective_length("a"), 1)
        # "A123" has length 4, no bonus for 1 capital = 4
        self.assertEqual(self.extractor.effective_length("A123"), 4)

        # "AB123" has length 5, +1 for 2 capitals = 6
        self.assertEqual(self.extractor.effective_length("AB123"), 6)
        # "ABC123" has length 6, +1 for 3 capitals = 7
        self.assertEqual(self.extractor.effective_length("ABC123"), 7)

        # "ABCD123" has length 7, +2 for 4 capitals = 9
        self.assertEqual(self.extractor.effective_length("ABCD123"), 9)
        # "ABCDEF123" has length 9, +2 for 6 capitals = 11
        self.assertEqual(self.extractor.effective_length("ABCDEF123"), 11)

    def test_combined_rules(self):
        """Test combinations of all rules"""
        # "A-B/CCCC000" has:
        # - 10 characters
        # - 2 deductions for dash/slash
        # - 3 deductions for repetitions (2 for CCCC, 1 for 000)
        # - +1 bonus for 3 capital letters
        # = 6 effective length
        self.assertEqual(self.extractor.effective_length("A-B/CCCC000"), 8)

        # "ABCDE-FG/HHHH00000" has:
        # - 18 characters
        # - 2 deductions for dash/slash
        # - 5 deductions for repetitions (2 for HHHH, 3 for 00000)
        # - +2 bonus for 7 capital letters
        # = 13 effective length
        self.assertEqual(self.extractor.effective_length("ABCDE-FG/HHHH00000"), 13)

    def test_edge_cases(self):
        """Test edge cases"""
        # "AAAAAAAA" has length 8, minus 6 for repetitions = 2, +1 for 8 capitals = 3
        self.assertEqual(self.extractor.effective_length("AAAAAAAA"), 4)

        # "aaaaaaaa" has length 8, minus 6 for repetitions = 2, no capital bonus = 2
        self.assertEqual(self.extractor.effective_length("aaaaaaaa"), 2)

        # Only dashes and slashes: length 6, minus 6 for special chars = 0
        self.assertEqual(self.extractor.effective_length("-/-/-/"), 0)

        # Special chars and repetitions: 9 chars, minus 3 for special chars, minus 6 for repetitions = 0
        self.assertEqual(self.extractor.effective_length("---///000"), 0)

        # "A0001234BBBB" has:
        # - 12 characters
        # - 3 deductions for repetitions (1 for 000, 2 for BBBB)
        # - +2 bonus for 4 capital letters
        # = 11 effective length
        self.assertEqual(self.extractor.effective_length("A0001234BBBB"), 11)


if __name__ == "__main__":
    unittest.main()
