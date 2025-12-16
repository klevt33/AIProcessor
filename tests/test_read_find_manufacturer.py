import asyncio

from config import Config
from matching_utils import find_manufacturers_in_description, read_manufacturer_data
from sdp import SDP
from utils import clean_description

# def find_manufacturer_excel():
#     manufacturer_dict = read_manufacturer_data(r"C:\Users\kirill.levtov\OneDrive - Perficient, Inc\Projects\ArchKey\Dev\Data\MFR_MAPPING.xlsx")
#     part_description = "3XLOGIC XYZ 12345 ACME ELECTRIC - DIVISION OF HUBBELL 1X2FT ABC567 "
#     matches = find_manufacturers_in_description(manufacturer_dict, part_description)
#     print(matches)


async def find_manufacturer(part_description):
    # Initialize configuration and SDP
    config = Config()
    sdp = SDP(config)

    # Call the async read_manufacturer_data function with sdp instance
    manufacturer_dict = await read_manufacturer_data(sdp)

    # Test with a sample part description
    clean_part_description = clean_description(part_description)
    matches = find_manufacturers_in_description(manufacturer_dict, clean_part_description)
    print(matches)


async def run_all_tests():
    await find_manufacturer("**3XLogic_XYZ_12345 ^ACME ELECTRIC -  DIVISION OF HUBBELL 1X2FT ABC567 3Mm.")
    await find_manufacturer("AFL_AFF_afi _XYZ_12345 ^ACME - ELECTRIC 1X2FT ABC567 TEST Param 1X2X3")
    await find_manufacturer("Alcoo ALLOY LLC DIVISION OF HUBBELL 1234567890 **33M** 1X2FT ABC567_")


# Create and run the async event loop
def main():
    asyncio.run(run_all_tests())


if __name__ == "__main__":
    main()

# class TestManufacturerMatching(unittest.TestCase):
#     def setUp(self):
#         self.config = Config()
#         self.sdp = SDP(self.config)

#     @patch('matching_utils.read_manufacturer_data')
#     async def test_find_manufacturer(self, mock_read_manufacturer_data):
#         # Mock the manufacturer dictionary
#         mock_manufacturer_dict = await read_manufacturer_data(self.sdp)
#         mock_read_manufacturer_data.return_value = mock_manufacturer_dict

#         test_cases = [
#             {
#                 "description": "**3XLogic_XYZ_12345 ^ACME ELECTRIC -  DIVISION OF HUBBELL 1X2FT ABC567 3Mm.",
#                 "expected": [
#                     ('ACME ELECTRIC - DIVISION OF HUBBELL', 'ACME ELECTRIC'),
#                     ('3MM', '3MM'),
#                     ('3XLOGIC', '3XLOGIC')
#                 ]
#             },
#             {
#                 "description": "AFL_AFF_afi _XYZ_12345 ^ACME - ELECTRIC 1X2FT ABC567 TEST Param 1X2X3",
#                 "expected": [
#                     ('ACME', 'ACME ELECTRIC'),
#                     ('AFF', 'ATKORE STEEL COMPONENTS'),
#                     ('AFI', 'AMERICAN FITTINGS CORPORATION')
#                 ]
#             },
#             {
#                 "description": "Alcoo ALLOY LLC DIVISION OF HUBBELL 1234567890 **** 1X2FT ABC567_",
#                 "expected": [('HUBBELL', 'HUBBELL')]
#             }
#         ]

#         for case in test_cases:
#             clean_part_description = clean_description(case["description"])
#             matches = find_manufacturers_in_description(mock_manufacturer_dict, clean_part_description)
#             self.assertEqual(matches, case["expected"])

# def run_async_test(coro):
#     return asyncio.run(coro)

# if __name__ == '__main__':
#     unittest.main()
