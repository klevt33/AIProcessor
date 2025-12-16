import re


def test_regex():
    # 1. The input string exactly as provided
    # Note: We use single quotes for the python string so the double quotes inside don't need escaping
    text = 'ACME L3300T-10-PG HALF SLOT 13/16" X 1-5/8" STRUT 12GA 10\' CHANNEL PRE GALVANIZED [BESC1316X10X12GASLZG90]'

    # 2. The regex pattern I suggested
    # Note: We use single quotes r'...' so the double quote " inside is treated as a literal character
    pattern = r'\b\d+(?:-\d+)?/\d+(?:-)?(?:IN\b|")'

    print("--- Debug Info ---")
    print(f"Input Text: {text}")
    print(f"Pattern:    {pattern}")

    # 3. Test finding matches
    matches = re.findall(pattern, text)
    print(f"\nMatches Found: {matches}")

    # 4. Test exclusion (substitution)
    cleaned_text = re.sub(pattern, "", text)
    # Clean up double spaces created by removal
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    print(f"\nOriginal: {text}")
    print(f"Cleaned:  {cleaned_text}")

    # Validation
    if '13/16"' in matches:
        print('\n✅ SUCCESS: 13/16" was matched.')
    else:
        print('\n❌ FAILED: 13/16" was NOT matched.')


if __name__ == "__main__":
    test_regex()
