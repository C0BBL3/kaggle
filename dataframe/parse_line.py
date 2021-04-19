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