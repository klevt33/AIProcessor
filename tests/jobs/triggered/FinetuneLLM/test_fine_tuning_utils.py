import unittest

from jobs.triggered.FinetuneLLM.fine_tuning_utils import FineTuningUtils

# Import the functions or classes you want to test
# from fine_tuning_utils import your_function_or_class


class TestFineTuningUtils(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures, if any."""

    def test_get_deployment_date_returns_correct_value(self):
        date = FineTuningUtils.get_deployment_date("gpt-4o-2024-08-06-spend-report-finetuned-08-07-25-dev")
        self.assertEqual("08-07-25", date)

    def test_get_all_old_deployment_names_returns_correct_values(self):
        all_deployment_names = [
            "gpt-4o-2024-08-06-spend-report-finetuned-08-07-25-dev",
            "gpt-4o-2024-08-06-spend-report-finetuned-08-07-25-prod",
            "gpt-4o-2024-08-06-spend-report-finetuned-6-11-25-dev",
            "gpt-4o-2024-08-06-spend-report-finetuned-6-11-25-prod",
        ]
        old_deployment_names = FineTuningUtils.get_all_old_deployment_names(all_deployment_names, n=1)
        self.assertEqual(2, len(old_deployment_names))
        self.assertEqual(
            ["gpt-4o-2024-08-06-spend-report-finetuned-6-11-25-dev", "gpt-4o-2024-08-06-spend-report-finetuned-6-11-25-prod"],
            old_deployment_names,
        )


if __name__ == "__main__":
    unittest.main()
