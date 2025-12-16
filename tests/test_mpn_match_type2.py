#!/usr/bin/env python3
"""
Test script to validate the behavior of the mpn_match_type function
from matching_scores.py, particularly for cases where part numbers appear
within larger strings with/without word boundaries.
"""

from typing import Dict, List, Tuple

from matching_scores import mpn_match_type


def test_mpn_match_type():
    """Test the mpn_match_type function with various scenarios."""

    test_cases: List[Tuple[str, str, int, str]] = [
        # Format: (part_num, description, expected_result, explanation)
        # Case 1: The primary scenario we're testing - part number within larger string
        ("12345", "XYZ-12345ABC", -1, "Part number without boundaries on both sides"),
        # Case 2: Exact match with boundaries on both sides
        ("12345", "XYZ 12345 ABC", 1, "Part number with boundaries on both sides"),
        # Case 3: Partial match with left boundary only
        ("12345", "XYZ 12345ABC", 0, "Part number with left boundary only"),
        # Case 4: Partial match with right boundary only
        ("12345", "XYZ12345 ABC", 0, "Part number with right boundary only"),
        # Case 5: Part number not in description at all
        ("12345", "This description does not contain the part", -1, "Part number not in description"),
        # Case 6: Description contains part number with separators
        ("12345", "XYZ 123-45 ABC", 1, "Part number with separators and boundaries"),
        # Case 7: Part number contains separators
        ("123-45", "XYZ 12345 ABC", 1, "Part number with separators, matching clean version with boundaries"),
        # Case 8: Part number at start of description
        ("12345", "12345 is at the start", 1, "Part number at start with right boundary"),
        # Case 9: Part number at end of description
        ("12345", "Part ending with 12345", 1, "Part number at end with left boundary"),
    ]

    results: Dict[str, List[Tuple[str, str, int, int, bool]]] = {"pass": [], "fail": []}

    print("\n=== Testing mpn_match_type function ===\n")

    for part_num, description, expected, explanation in test_cases:
        actual = mpn_match_type(part_num, description)
        status = actual == expected

        result_category = "pass" if status else "fail"
        results[result_category].append((part_num, description, expected, actual, status))

        status_str = "✓" if status else "✗"
        print(f"{status_str} Test: {explanation}")
        print(f"   Part: '{part_num}' in '{description}'")
        print(f"   Expected: {expected}, Actual: {actual}")
        print()

    # Summary
    total = len(test_cases)
    passed = len(results["pass"])
    failed = len(results["fail"])

    print("\n=== Test Summary ===")
    print(f"Total tests:  {total}")
    print(f"Passed:       {passed}")
    print(f"Failed:       {failed}")

    if failed > 0:
        print("\n=== Failed Tests ===")
        for part_num, description, expected, actual, _ in results["fail"]:
            print(f"Part: '{part_num}' in '{description}'")
            print(f"Expected: {expected}, Actual: {actual}")
            print()

    # Specifically analyze the original scenario
    original_case = test_cases[0]
    print("\n=== Analysis of Original Scenario ===")
    print(f"Part: '{original_case[0]}' in '{original_case[1]}'")
    print(f"Expected: {original_case[2]} (based on your understanding)")
    actual = mpn_match_type(original_case[0], original_case[1])
    print(f"Actual: {actual}")

    if actual != original_case[2]:
        print("\nDiscrepancy found!")
        if actual == 0:
            print("The function returns 0 (partial boundary match) rather than -1 for this case.")
            print("This suggests the function is treating 'XYZ-12345ABC' as having at least one boundary.")
        else:
            print(f"The function returns {actual} for this case, which contradicts both interpretations.")

        print("\nThis indicates that either:")
        print("1. The function is not correctly implementing the intended logic, or")
        print("2. The documentation's interpretation of the return values is incorrect.")


if __name__ == "__main__":
    test_mpn_match_type()
