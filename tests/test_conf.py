import unittest


# copied from the main code since it is an inner function to test
def get_tokens_of_string(token_list, value, strip_chars=' \n"'):
    """
    Finds the contiguous token index range in token_list whose cleaned
    concatenation equals `value`, allowing the last token to overhang,
    and only stripping unwanted chars at the *start* of each token.

    Args:
    token_list   (list[str]): raw tokens from the model
    value        (str): exact target substring to match
    strip_chars (str): characters to strip from the *start* of first token and *end* of last token

    Returns:
    List[int]: indices [start, …, end] of the matching tokens, or [] if none.
    """
    for start in range(len(token_list)):
        clean_token = token_list[start].lstrip(strip_chars).encode().decode("unicode_escape")
        clean_concat = clean_token
        if clean_token.rstrip(strip_chars) == value:
            return [start]
        if not value.startswith(clean_concat) or not clean_concat:
            continue
        for end in range(start + 1, len(token_list)):
            raw_tok = token_list[end].encode().decode("unicode_escape")
            # what remains of `value` after what we've concatenated so far?
            remaining = value[len(clean_concat) :]

            if remaining.startswith(raw_tok):
                # No need to clean
                if raw_tok.startswith(remaining):
                    # Found the final piece
                    return list(range(start, end + 1))
                clean_concat += raw_tok
            else:
                # only remove leading noise; keep trailing punctuation intact
                clean_tok = raw_tok.rstrip(strip_chars)
                # skip tokens that become empty once cleaned
                if not clean_tok:
                    continue

                # 1) if clean_tok starts with exactly that remaining piece, we're done
                if clean_tok.startswith(remaining):
                    return list(range(start, end + 1))

                # 2) otherwise append and keep matching
                break

    return []


class TestGetTokensOfString(unittest.TestCase):
    def test_failed_match_1(self):
        token_list = [
            "```",
            "json",
            "  \n",
            "{",
            "  \n",
            "   ",
            ' "',
            "Manufacturer",
            "Name",
            '":',
            ' "',
            "LOT",
            "US",
            " LED",
            " LIGHT",
            "S",
            '",',
            "  \n",
            "   ",
            ' "',
            "Part",
            "Number",
            '":',
            ' "',
            "Q",
            "TRAN",
            "2",
            "ND",  # codespell:ignore
            "RE",
            "LEASE",
            "80",
            "TYPE",
            "L",
            "34",
            "LV",
            "L",
            "12",
            "AL",
            "TA",
            "01",
            "SW",
            "4",
            ".",
            "027",
            "DR",
            "Y",
            "DFS",
            "2",
            "BW",
            "24",
            "CLS",
            "WH",
            "CL",
            "2",
            "MG",
            "ST",
            "O",
            "XX",
            '\\"',
            "FEET ",
        ]
        value = 'QTRAN2NDRELEASE80TYPEL34LVL12ALTA01SW4.027DRYDFS2BW24CLSWHCL2MGSTOXX"FEET'
        self.assertNotEqual(get_tokens_of_string(token_list, value), [])

    def test_failed_match_2(self):
        token_list = [
            "```",
            "json",
            "  \n",
            "{",
            "  \n",
            "   ",
            ' "',
            "Manufacturer",
            "Name",
            '":',
            ' "',
            "STR",
            "UT",
            " FAST",
            '",',
            "  \n",
            "   ",
            ' "',
            "Part",
            "Number",
            '":',
            ' "',
            "158",
            '\\"',
            "X",
            "158",
            '\\"',
            "X",
            "10",
            "FT",
            "LONG",
            "12",
            "GA",
            "UG",
            "ES",
            "LO",
            "TT",
            "ED",
            "P",
            "REG",
            "AL",
            "V",
            "CHANNEL",
            "FS",
            "200",
            "SS",
            "PG",
            "10",
            "B",
            "22",
            "SH",
            "120",
            "GL",
            "V",
            '",',
            "  \n",
            "   ",
            ' "',
            "UN",
        ]
        value = '158"X158"X10FTLONG12GAUGESLOTTEDPREGALVCHANNELFS200SSPG10B22SH120GLV'
        self.assertNotEqual(get_tokens_of_string(token_list, value), [])

    def test_failed_match_3(self):
        token_list = [
            "```",
            "json",
            "  \n",
            "{",
            "  \n",
            "   ",
            ' "',
            "Manufacturer",
            "Name",
            '":',
            ' "',
            "GEN",
            "ERIC",
            " ELECT",
            "R",
            "ICAL",
            " PRODUCT",
            '",',
            "  \n",
            "   ",
            ' "',
            "Part",
            "Number",
            '":',
            ' "',
            "2",
            '\\"',
            "x",
            "15",
            "'",
            "F",
            "LEX",
            "RIS",
            "ER",
            '",',
            "  \n",
            "   ",
            ' "',
            "UN",
            "S",
            "PSC",
            '":',
            ' "',
            "391",
            "217",
            "25",
            '"',
            "  \n",
            "}",
            "  \n",
            "```",
        ]
        value = "2\"x15'FLEXRISER"
        self.assertNotEqual(get_tokens_of_string(token_list, value), [])


if __name__ == "__main__":
    unittest.main()
