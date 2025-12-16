import sys
import types
import unittest
from unittest.mock import Mock

# Patch missing module before importing the class
sys.modules["finetune_constants"] = types.ModuleType("finetune_constants")

# Create a mock context with a mock logger
mock_context = Mock()
mock_context.logger = Mock()

# Create dummy app_context with a dummy 'context' attribute
mock_app_context = types.ModuleType("app_context")
mock_app_context.context = mock_context
sys.modules["app_context"] = mock_app_context

from jobs.triggered.FinetuneLLM.matcher import Matcher


class TestMatcher(unittest.IsolatedAsyncioTestCase):
    async def test_check_mfr_pn_name_match_returns_yes(self):
        result = await Matcher._check_mfr_pn_name_match("!@#$%^&*)()predicted value", "predicted value")
        self.assertEqual("yes", result.value)
        result = await Matcher._check_mfr_pn_name_match("predicted value", "!@#$%^&*)()predicted value")
        self.assertEqual("yes", result.value)
        result = await Matcher._check_mfr_pn_name_match("value", "predicted value")
        self.assertEqual("yes", result.value)

    async def test_check_mfr_pn_name_match_returns_no(self):
        result = await Matcher._check_mfr_pn_name_match("expected value", "predicted value")
        self.assertEqual("no", result.value)

    async def test_check_mfr_pn_name_match_returns_skip_on_empty_expected_string(self):
        result = await Matcher._check_mfr_pn_name_match("", "predicted value")
        self.assertEqual("skip", result.value)

    async def test_prefix_or_suffix_match_returns_False(self):
        result = Matcher._prefix_or_suffix_match("this is a test", "is a")
        self.assertFalse(result)
        result = Matcher._prefix_or_suffix_match("this is a test", "test ")
        self.assertFalse(result)
        result = Matcher._prefix_or_suffix_match("testt", "this is a test")
        self.assertFalse(result)
        result = Matcher._prefix_or_suffix_match("this is aa", "this is a test")
        self.assertFalse(result)

    async def test_prefix_or_suffix_match_returns_True(self):
        result = Matcher._prefix_or_suffix_match("this is a test", "test")
        self.assertTrue(result)
        result = Matcher._prefix_or_suffix_match("this is a test", "this is a")
        self.assertTrue(result)
        result = Matcher._prefix_or_suffix_match("test", "this is a test")
        self.assertTrue(result)
        result = Matcher._prefix_or_suffix_match("this is a", "this is a test")
        self.assertTrue(result)

    async def test_check_unspsc_match_returns_no(self):
        result = await Matcher._check_unspsc_match("32345678", "12345690")
        self.assertEqual("no", result.value)
        result = await Matcher._check_unspsc_match("12345678", "12345378")
        self.assertEqual("no", result.value)
        result = await Matcher._check_unspsc_match("12000000", "12000001", digits=8)
        self.assertEqual("no", result.value)

    async def test_check_unspsc_match_returns_skip(self):
        result = await Matcher._check_unspsc_match("-", "12345690", digits=6)
        self.assertEqual("skip", result.value)
        result = await Matcher._check_unspsc_match("", "12345678", digits=6)
        self.assertEqual("skip", result.value)

    async def test_check_unspsc_match_returns_yes(self):
        result = await Matcher._check_unspsc_match("12345678", "12345690", digits=6)
        self.assertEqual("yes", result.value)
        result = await Matcher._check_unspsc_match("12345678", "12345678", digits=6)
        self.assertEqual("yes", result.value)
        result = await Matcher._check_unspsc_match("12000000", "12111111", digits=2)
        self.assertEqual("yes", result.value)
