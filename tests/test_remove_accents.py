# test_remove_accents.py

import unittest

from utils import remove_accents


class TestRemoveAccents(unittest.TestCase):
    """
    Unit tests for the hardened remove_accents function in utils.py.
    This test suite is designed to be robust against file encoding issues
    by using Unicode escapes for special characters where necessary.
    """

    def test_empty_string(self):
        """Test that an empty string returns an empty string."""
        self.assertEqual(remove_accents(""), "")

    def test_string_with_no_accents(self):
        """Test a regular ASCII string remains unchanged."""
        self.assertEqual(remove_accents("hello world 123"), "hello world 123")

    def test_string_with_common_symbols(self):
        """Test that numbers and common symbols are not affected."""
        self.assertEqual(remove_accents("test-123!@#$%^&*()"), "test-123!@#$%^&*()")

    def test_french_characters(self):
        """Test common French accented characters."""
        original = "voilà une crème brûlée, garçon. C'est Noël !"
        expected = "voila une creme brulee, garcon. C'est Noel !"
        self.assertEqual(remove_accents(original), expected)

    def test_german_characters(self):
        """Test German umlauts and the eszett (ß) on a real word."""
        # We use the unicode escape \u00df for ß to avoid file encoding issues.
        original = "Die Goethestra\u00dfe is in München."
        expected = "Die Goethestrasse is in Munchen."
        self.assertEqual(remove_accents(original), expected)

    def test_spanish_characters(self):
        """Test Spanish accented characters and ñ."""
        original = "El pingüino se comió el jalapeño rápidamente."
        expected = "El pinguino se comio el jalapeno rapidamente."
        self.assertEqual(remove_accents(original), expected)

    def test_nordic_characters(self):
        """Test characters from Nordic languages using Unicode escapes."""
        # We use \u00f8 for ø to avoid file encoding issues.
        original = "Sm\u00f8rrebr\u00f8d and Ångström"
        expected = "Smorrebrod and Angstrom"
        self.assertEqual(remove_accents(original), expected)

    def test_icelandic_characters(self):
        """Test Icelandic characters Thorn (Þ, þ) and Eth (Ð, ð)."""
        # Unicode escapes: Þ=\u00de, þ=\u00fe, Ð=\u00d0, ð=\u00f0
        original = "Þór hafði það gott með Guðrúnu á Ísafirði."
        expected = "Thor hafdi thad gott med Gudrunu a Isafirdi."
        self.assertEqual(remove_accents(original), expected)

    def test_mixed_language_string(self):
        """Test a string containing characters from multiple languages."""
        # Note: 'frábært' contains 'æ' which must be converted to 'ae'.
        original = "Pýthöñ is âwesome, Jörg! Það er frábært."
        expected = "Python is awesome, Jorg! Thad er frabaert."
        self.assertEqual(remove_accents(original), expected)

    def test_non_latin_characters_are_removed(self):
        """Test that characters from other scripts (e.g., Greek) are removed."""
        original = "Hello Ελληνικά and Русский world"
        expected = "Hello  and  world"
        self.assertEqual(remove_accents(original), expected)

    def test_emojis_are_removed(self):
        """Test that emojis are stripped from the string."""
        original = "Python is fun! 🎉🐍"
        expected = "Python is fun! "
        self.assertEqual(remove_accents(original), expected)


# This allows the test to be run from the command line
if __name__ == "__main__":
    unittest.main(verbosity=2)
