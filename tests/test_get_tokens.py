# fmt: off
import pytest

from ai_utils import _get_tokens_of_string


@pytest.mark.parametrize("value,token_list",
                         [
                             ('TYPE A4/A5 40DEG GRAZER LENS', ['```', 'json', '\n', ' ', ' {\n', ' ', ' "', 'Manufacturer', 'Name', '":', ' "', 'CO', 'OPER', ' LIGHT', 'ING', '",\n', ' ', ' "', 'Part', 'Number', '":', ' "', 'TYPE', ' A', '4', '/A', '5', ' ', '40', 'DE', 'G', ' G', 'RA', 'ZER', ' L', 'ENS', '",\n', ' ', ' "', 'UN', 'S', 'PSC', '":', ' "', '391', '115', '41', '"\n', ' ', ' }\n', ' ', ' ```']),
                             ('WLTE W 1 G M6', ['```', 'json', '\n', ' ', ' {\n', ' ', ' "', 'Manufacturer', 'Name', '":', ' "', 'L', 'ITH', 'ON', 'IA', ' LIGHT', 'ING', '",\n', ' ', ' "', 'Part', 'Number', '":', ' "', 'WL', 'TE', ' W', ' ', '1', ' G', ' M', '6', '",\n', ' ', ' "', 'UN', 'S', 'PSC', '":', ' "', '391', '117', '08', '"\n', ' ', ' }\n', ' ', ' ```']),
                             ('PRO XD MID-SIZE 11449089', ['```', 'json', '\n', ' ', ' {\n', ' ', ' "', 'Manufacturer', 'Name', '":', ' "', 'POL', 'AR', 'IS', '",\n', ' ', ' "', 'Part', 'Number', '":', ' "', 'PRO', ' XD', ' MID', '-S', 'IZE', ' ', '114', '490', '89', '",\n', ' ', ' "', 'UN', 'S', 'PSC', '":', ' "', '251', '017', '09', '"\n', ' ', ' }\n', ' ', ' ```']),
                         ])
def test_find_some_tokens(token_list, value):
    result = _get_tokens_of_string(token_list, value)
    assert len(result) > 0


@pytest.mark.parametrize("value,token_list",
                         [
                             ('TYPEEEEE A4/A5 40DEG GRAZER LENS', ['```', 'json', '\n', ' ', ' {\n', ' ', ' "', 'Manufacturer', 'Name', '":', ' "', 'CO', 'OPER', ' LIGHT', 'ING', '",\n', ' ', ' "', 'Part', 'Number', '":', ' "', 'TYPE', ' A', '4', '/A', '5', ' ', '40', 'DE', 'G', ' G', 'RA', 'ZER', ' L', 'ENS', '",\n', ' ', ' "', 'UN', 'S', 'PSC', '":', ' "', '391', '115', '41', '"\n', ' ', ' }\n', ' ', ' ```']),
                             ('WLTE WW 1 G M6', ['```', 'json', '\n', ' ', ' {\n', ' ', ' "', 'Manufacturer', 'Name', '":', ' "', 'L', 'ITH', 'ON', 'IA', ' LIGHT', 'ING', '",\n', ' ', ' "', 'Part', 'Number', '":', ' "', 'WL', 'TE', ' W', ' ', '1', ' G', ' M', '6', '",\n', ' ', ' "', 'UN', 'S', 'PSC', '":', ' "', '391', '117', '08', '"\n', ' ', ' }\n', ' ', ' ```']),
                             ('PRO XD MID-SIZE 11449089', ['```', 'json', '\n', ' ', ' {\n', ' ', ' "', 'Manufacturer', 'Name', '":', ' "', 'POL', 'AR', 'IS', '",\n', ' ', ' "', 'Part', 'Number', '":', ' "', 'PRO', ' XD', ' MID RANGE', '-S', 'IZE', ' ', '114', '490', '89', '",\n', ' ', ' "', 'UN', 'S', 'PSC', '":', ' "', '251', '017', '09', '"\n', ' ', ' }\n', ' ', ' ```']),
                         ])
def test_doesnt_find_tokens_when_doesnt_exist(token_list, value):
    result = _get_tokens_of_string(token_list, value)
    assert len(result) == 0
