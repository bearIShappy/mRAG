import re
from typing import List


def extract_points(text: str) -> List[str]:
    """
    Extract numbered points like:
    1. text
    2. text
    """

    extracted_matches = []

    pattern = r'\d+\.\s(.*?)(?=\n\d+\.\s|$)'
    matches = re.findall(pattern, text, re.DOTALL)

    for match in matches:
        extracted_matches.append(match.strip())

    if not extracted_matches:
        return [text]

    return extracted_matches