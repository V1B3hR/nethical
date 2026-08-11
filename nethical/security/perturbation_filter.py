"""Input Perturbation Filter.

Cleanses and normalizes input queries to neutralize adversarial modifications
and evasion techniques before evaluation.
"""

import re
import unicodedata

class InputPerturbationFilter:
    """Filters and sanitizes inputs to neutralize adversarial perturbations."""

    @staticmethod
    def sanitize(text: str) -> str:
        """Sanitize a text payload using homoglyph mapping, whitespace normalization, and control character stripping.

        Args:
            text: Raw input string

        Returns:
            Sanitized, normalized string
        """
        if not text:
            return ""

        # 1. Remove suspicious hidden tags/comments used in indirect injections (like html comments or system prompts within delimiters)
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

        # 2. Strip non-printable / control characters & zero-width characters
        # Matches zero-width spaces, invisible separators, control characters
        text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\r\t")

        # 3. Normalize Unicode form (NFKC matches compatibility forms, decomposing homoglyphs)
        text = unicodedata.normalize("NFKC", text)

        # 4. Collapse excessive consecutive whitespaces/newlines
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()
