from azure_search_utils import AzureSearchUtils
from config import Config
from llm import LLM
from semantic_matching import semantic_match_by_description
from utils import clean_description

# Initialize required objects
config = Config()
search_utils = AzureSearchUtils(config)
llm = LLM(config)

# Define test scenarios
test_scenarios = [
    # # Exact Matches
    # {
    #     "original_description": "AUS AB-101048GST 10X10X48 SC THROUGH",
    #     "modified_description": "AUS AB-101048GST 10X10X48 SC THROUGH",
    #     "expected_mfr_name": "AUSTIN ELECTRICAL ENCLOSURES",
    #     "expected_unspsc": "39131714"
    # },
    # {
    #     "original_description": "MILW 48-22-9486 RATCHET/SOCKET SET",
    #     "modified_description": "MILW 48-22-9486 RATCHET SOCKET SET",
    #     "expected_mfr_name": "MILWAUKEE TOOL",
    #     "expected_unspsc": "27111753"
    # },
    # # Slightly Modified Descriptions
    # {
    #     "original_description": "CRL 01480.41T.05 12AWG CORD",
    #     "modified_description": "CRL 01480.41T.05 12 AWG CORD",
    #     "expected_mfr_name": "CAROLINA PRODUCTS",
    #     "expected_unspsc": "26121545"
    # },
    # {
    #     "original_description": "KLEIN 603-10 NO-2 PHILLIPS SCR-DRVR",
    #     "modified_description": "KLEIN 603-10 PHILLIPS SCREWDRIVER NO-2",
    #     "expected_mfr_name": "KLEIN TOOLS",
    #     "expected_unspsc": "27111701"
    # },
    # # Moderately Modified Descriptions
    # {
    #     "original_description": "HOFF A84XM3EW24FTC ENCLOSURE",
    #     "modified_description": "HOFF A84XM3EW24FTC",
    #     "expected_mfr_name": "HOFFMAN",
    #     "expected_unspsc": "39121318"
    # },
    # {
    #     "original_description": "MILW 48-13-8275 3/4X54 CABLE BIT",
    #     "modified_description": "MILW 48-13-8275 CABLE BIT",
    #     "expected_mfr_name": "MILWAUKEE TOOL",
    #     "expected_unspsc": "27112700"
    # },
    # # Descriptions with Typos or Minor Errors
    # {
    #     "original_description": "PAND PN10-6R-L #10 NYL RG TERM",
    #     "modified_description": "PAND PN10-6R-L #10 NYLON RING TERMINAL",
    #     "expected_mfr_name": "PANDUIT",
    #     "expected_unspsc": "39121432"
    # },
    # {
    #     "original_description": "CRC 02120 20OZ DEGREASER",
    #     "modified_description": "CRC 02120 20 OZ DEGREESER",  # Typo: "DEGREESER" instead of "DEGREASER"
    #     "expected_mfr_name": "CRC INDUSTRIES",
    #     "expected_unspsc": "47131821"
    # },
    # {
    #     "original_description": "DOTTIE MB58334 MACHINE BOLT",
    #     "modified_description": "DOTTIE MB58334 MACHNE BOLT",  # Typo: "MACHNE" instead of "MACHINE"
    #     "expected_mfr_name": "LH DOTTIE",
    #     "expected_unspsc": "31161620"
    # },
    # {
    #     "original_description": "KLEIN D318-51/2C NEEDLE NOSE PLIER",
    #     "modified_description": "KLEIN D318-51/2C NEEDLE NOSE PLIERS",  # Pluralization
    #     "expected_mfr_name": "KLEIN TOOLS",
    #     "expected_unspsc": "27112108"
    # },
    # # Slightly Modified Descriptions
    # {
    #     "original_description": "AUS AB-101048GST 10X10X48 SC THROUGH",
    #     "modified_description": "SC THROUGH AUS AB-101048GST 10X10X48",  # Reordered words
    #     "expected_mfr_name": "AUSTIN ELECTRICAL ENCLOSURES",
    #     "expected_unspsc": "39131714"
    # },
    # {
    #     "original_description": "MILW 48-22-9486 RATCHET/SOCKET SET",
    #     "modified_description": "MILWAUKEE TOOL 48-22-9486 RATCHET SOCKET SET",  # Added manufacturer name
    #     "expected_mfr_name": "MILWAUKEE TOOL",
    #     "expected_unspsc": "27111753"
    # },
    # {
    #     "original_description": "KLEIN D318-51/2C NEEDLE NOSE PLIER",
    #     "modified_description": "KLEIN TOOLS D318-51/2C NEEDLE NOSE PLIERS",  # Pluralization and added brand
    #     "expected_mfr_name": "KLEIN TOOLS",
    #     "expected_unspsc": "27112108"
    # },
    # # Moderately Modified Descriptions
    # {
    #     "original_description": "CRC 02120 20OZ DEGREASER",
    #     "modified_description": "CRC 02120 DEGREASER",  # Removed size
    #     "expected_mfr_name": "CRC INDUSTRIES",
    #     "expected_unspsc": "47131821"
    # },
    # {
    #     "original_description": "DOTTIE MB58334 MACHINE BOLT",
    #     "modified_description": "MB58334 BOLT",  # Shortened description
    #     "expected_mfr_name": "LH DOTTIE",
    #     "expected_unspsc": "31161620"
    # },
    # {
    #     "original_description": "PAND PN10-6R-L #10 NYL RG TERM",
    #     "modified_description": "PN10-6R-L TERMINAL",  # Removed manufacturer and material
    #     "expected_mfr_name": "PANDUIT",
    #     "expected_unspsc": "39121432"
    # },
    # # Descriptions with Typos or Minor Errors
    # {
    #     "original_description": "CFI 25DXW81P 2-1/2 45D EL",
    #     "modified_description": "CFI 25DXW81P 2-1/2 45D ELBOW",  # Corrected abbreviation
    #     "expected_mfr_name": "CHAMPION FIBERGLASS",
    #     "expected_unspsc": "39131700"
    # },
    # {
    #     "original_description": "MILW 48-13-8275 3/4X54 CABLE BIT",
    #     "modified_description": "MILW 48-13-8275 3/4X54 CABBLE BIT",  # Typo: "CABBLE" instead of "CABLE"
    #     "expected_mfr_name": "MILWAUKEE TOOL",
    #     "expected_unspsc": "27112700"
    # },
    # {
    #     "original_description": "ZSI L310000S4 4HL CORNER ANGLE",
    #     "modified_description": "ZSI L310000S4 4HL CORNER ANGEL",  # Typo: "ANGEL" instead of "ANGLE"
    #     "expected_mfr_name": "ZSI FOSTER MANUFACTURING",
    #     "expected_unspsc": "30101505"
    # },
    # # Descriptions with Added Noise or Irrelevant Information
    # {
    #     "original_description": "HOFF A84XM3EW24FTC ENCLOSURE",
    #     "modified_description": "HOFF A84XM3EW24FTC ENCLOSURE FOR OUTDOOR USE",  # Added irrelevant info
    #     "expected_mfr_name": "HOFFMAN",
    #     "expected_unspsc": "39121318"
    # },
    # {
    #     "original_description": "KLEIN 603-10 NO-2 PHILLIPS SCR-DRVR",
    #     "modified_description": "KLEIN 603-10 NO-2 PHILLIPS SCREWDRIVER FOR WOODWORKING",  # Added context
    #     "expected_mfr_name": "KLEIN TOOLS",
    #     "expected_unspsc": "27111701"
    # },
    # {
    #     "original_description": "CRC 02120 20OZ DEGREASER",
    #     "modified_description": "CRC 02120 20OZ DEGREASER - FAST ACTING",  # Added marketing text
    #     "expected_mfr_name": "CRC INDUSTRIES",
    #     "expected_unspsc": "47131821"
    # },
    # # Descriptions with Abbreviations or Missing Words
    # {
    #     "original_description": "PAND TP2-C TIE PLATE MOUNT",
    #     "modified_description": "TP2-C TIE PLATE",  # Removed manufacturer and mount
    #     "expected_mfr_name": "PANDUIT",
    #     "expected_unspsc": "39121700"
    # },
    # {
    #     "original_description": "MILB 1818FLC FRT FLSH CVR",
    #     "modified_description": "MILB 1818FLC FLUSH COVER",  # Removed "FRT" abbreviation
    #     "expected_mfr_name": "MILBANK MANUFACTURING",
    #     "expected_unspsc": "39121304"
    # },
    # {
    #     "original_description": "CRL 01480.41T.05 12AWG CORD",
    #     "modified_description": "CRL 01480.41T.05 12GA CORD",  # Abbreviation: "GA" instead of "AWG"
    #     "expected_mfr_name": "CAROLINA PRODUCTS",
    #     "expected_unspsc": "26121545"
    # },
    {
        "original_description": "OBIT - 4AR1G-58 1G 4S ADJ RING 5/8IN-1-1/4IN D",
        "modified_description": "OBIT - 4AR1G-58 1G 4S ADJ RING 5/8IN-1-1/4IN D",
        "expected_mfr_name": "WHO KNOWS",
        "expected_unspsc": "????????",
    }
]

# Run test scenarios
for scenario in test_scenarios:
    # Clean the modified description
    cleaned_description = clean_description(scenario["modified_description"])

    # Call semantic_match_by_description
    result = semantic_match_by_description(description=cleaned_description, azure_search_utils=search_utils, llm=llm)

    # Extract actual results
    actual_mfr_name = result.get("MfrName", None)
    actual_unspsc = result.get("UNSPSC", None)
    mfr_confidence_score = result.get("ManufacturerNameConfidenceScore", None)
    unspsc_confidence_score = result.get("UNSPSCConfidenceScore", None)

    # Compare UNSPSC values with flexibility
    expected_unspsc = scenario["expected_unspsc"]
    unspsc_match = False
    if actual_unspsc:
        if actual_unspsc == expected_unspsc:
            unspsc_match = True
        elif actual_unspsc.endswith("00") and actual_unspsc[:-2] == expected_unspsc[:-2]:
            # Match if the last 2 digits are "00" and the rest of the digits match
            unspsc_match = True
        elif actual_unspsc.endswith("0000") and actual_unspsc[:-4] == expected_unspsc[:-4]:
            # Match if the last 4 digits are "0000" and the rest of the digits match
            unspsc_match = True

    # Print results
    print(f"Original Description: {scenario['original_description']}")
    print(f"Modified Description: {scenario['modified_description']}")
    print(f"Expected Manufacturer: {scenario['expected_mfr_name']}, Actual: {actual_mfr_name}")
    print(f"Expected UNSPSC: {expected_unspsc}, Actual: {actual_unspsc} (Match: {unspsc_match})")
    print(f"Mfr Confidence Score: {mfr_confidence_score}")
    print(f"UNSPSC Confidence Score: {unspsc_confidence_score}")
    print("-" * 80)
