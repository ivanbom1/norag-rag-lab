"""Personal data redaction: patterns only (emails, phone numbers, IBANs, card numbers, IP addresses).

Text sent to the LLM provider and text written to traces goes through a Redactor first. Each value gets a numbered
placeholder ([EMAIL_1]), the same one every time it appears in the request, so the LLM can still refer to it.
restore() puts the values back in what the user sees; the mapping only lives in memory for one request.
"""
import re
from collections import Counter

PATTERNS = [
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b")),
    ("CARD", re.compile(r"\b\d(?:[ -]?\d){12,18}\b")),
    ("PHONE", re.compile(
        r"(?<![\w+])(?:\+\d{1,3}[ .-]?(?:\(0\)[ .-]?)?\d{1,4}(?:[ .-]?\d{2,4}){2,4}"  # international, +33 6 12 34 56 78
        r"|0[1-9](?:[ .-]?\d{2}){4}"  # French, 06 12 34 56 78
        r"|\(\d{3}\) ?\d{3}-\d{4}|\b\d{3}-\d{3}-\d{4})(?![\w-])")),  # North American
    # after PHONE, and not inside a longer dotted number, so 06.99.00.56.78 stays a phone number
    ("IP", re.compile(r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?!\.?\d)")),
]


def luhn_ok(number):
    digits = [int(d) for d in re.sub(r"\D", "", number)][::-1]
    return sum(d if i % 2 == 0 else (d * 2 - 9 if d > 4 else d * 2) for i, d in enumerate(digits)) % 10 == 0


def iban_ok(iban):
    compact = iban.replace(" ", "")
    rearranged = compact[4:] + compact[:4]
    return int("".join(str(int(c, 36)) for c in rearranged)) % 97 == 1


CHECKS = {"CARD": luhn_ok, "IBAN": iban_ok}


class Redactor:
    def __init__(self):
        self.placeholders = {}
        self.values = {}
        self.counts = Counter()

    def placeholder(self, kind, value):
        if value not in self.placeholders:
            self.counts[kind] += 1
            self.placeholders[value] = f"[{kind}_{self.counts[kind]}]"
            self.values[self.placeholders[value]] = value
        return self.placeholders[value]

    def redact(self, text):
        for kind, pattern in PATTERNS:
            check = CHECKS.get(kind, lambda value: True)
            text = pattern.sub(lambda m: self.placeholder(kind, m.group()) if check(m.group()) else m.group(), text)
        return text

    def restore(self, text):
        return re.sub(r"\[[A-Z]+_\d+\]", lambda m: self.values.get(m.group(), m.group()), text)
