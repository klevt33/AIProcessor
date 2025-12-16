import unittest

from utils import clean_description


class TestCleanDescription(unittest.TestCase):
    def test_uppercase_conversion(self):
        self.assertEqual(clean_description("hello world"), "HELLO WORLD")

    def test_special_characters_replacement(self):
        self.assertEqual(clean_description("hello_world|test*example="), "HELLO WORLD TEST EXAMPLE")

    def test_trailing_and_leading_removal(self):
        self.assertEqual(clean_description(" -hello- "), "HELLO")
        self.assertEqual(clean_description("...test..."), "TEST")

    def test_caret_removal(self):
        self.assertEqual(clean_description("he^llo^ world"), "HELLO WORLD")

    # def test_dash_spacing(self):
    #     self.assertEqual(clean_description("hello - world"), "HELLO-WORLD")
    #     self.assertEqual(clean_description("hello- world"), "HELLO-WORLD")
    #     self.assertEqual(clean_description("hello -world"), "HELLO-WORLD")

    def test_multiple_spaces(self):
        self.assertEqual(clean_description("hello    world"), "HELLO WORLD")
        self.assertEqual(clean_description("  multiple   spaces   test  "), "MULTIPLE SPACES TEST")


if __name__ == "__main__":
    unittest.main()
