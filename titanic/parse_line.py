def parse_line(line):
    entries = []
    entry_str = ""
    quote_symbol = None
    for current_character in line:
        if quote_symbol is None and current_character == ",":
            entries.append(entry_str)
            entry_str = ""
        else:
            entry_str += current_character
        if current_character == '\"' or current_character == "'" or current_character == "\'":
            if quote_symbol == current_character:
                quote_symbol = None
            elif quote_symbol is None:
                quote_symbol = current_character
                
    entries.append(entry_str)
    return entries
'''

print('\nTesting...\n')

print("    Testing parse_line")
assert parse_line("1,0,3,'Braund, Mr. Owen Harris',male,22,1,0,A/5 21171,7.25,,S") == ['1', '0', '3', "'Braund, Mr. Owen Harris'", 'male', '22', '1', '0', 'A/5 21171', '7.25', '', 'S'], "parse_line was not right, it should be ['1', '0', '3', "'Braund, Mr. Owen Harris'", 'male', '22', '1', '0', 'A/5 21171', '7.25', '', 'S'], but was {}".format(parse_line("1,0,3,'Braund, Mr. Owen Harris',male,22,1,0,A/5 21171,7.25,,S"))
print("    parse_line Passed!!!\n")

print("    Testing parse_line")
assert parse_line('102,0,3,"Petroff, Mr. Pastcho (""Pentcho"")",male,,0,0,349215,7.8958,,S') == ['102', '0', '3', '"Petroff, Mr. Pastcho (""Pentcho"")"', 'male', '', '0', '0', '349215', '7.8958', '', 'S'], "parse_line was not right, it should be ['102', '0', '3', ''Petroff, Mr. Pastcho ("'Pentcho'")'', 'male', '', '0', '0', '349215', '7.8958', '', 'S'], but was {}".format(parse_line('102,0,3,"Petroff, Mr. Pastcho (""Pentcho"")",male,,0,0,349215,7.8958,,S'))
print("    parse_line Passed!!!\n")

print("    Testing parse_line")
assert parse_line('187,1,3,"O\'Brien, Mrs. Thomas (Johanna ""Hannah"" Godfrey)",female,,1,0,370365,15.5,,Q') == ['187', '1', '3', '"O\'Brien, Mrs. Thomas (Johanna ""Hannah"" Godfrey)"', 'female', '', '1', '0', '370365', '15.5', '', 'Q'], "parse_line was not right, it should be ['187', '1', '3', ''O'Brien, Mrs. Thomas (Johanna ""Hannah"" Godfrey)'', 'female', '', '1', '0', '370365', '15.5', '', 'Q'], but was {}".format(parse_line(parse_line('187,1,3,"O\'Brien, Mrs. Thomas (Johanna ""Hannah"" Godfrey)",female,,1,0,370365,15.5,,Q')))
print("    parse_line Passed!!!\n")

print('ALL TESTS PASS!!!!!')'''