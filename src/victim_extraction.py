import re

def extract_victims(text):

    pattern = r'(\d+)\s*(people|persons|injured|dead|victims)'
    match = re.search(pattern, text.lower())

    if match:
        return int(match.group(1))

    return None