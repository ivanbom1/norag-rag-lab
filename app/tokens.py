import re

# Letters/digits joined by an inner - . ' or ’, so KST-2044, 203.0.113.10 and captain’s stay one word.
TOKEN = re.compile(r"[^\W_]+(?:[-.'’][^\W_]+)*")

STOPWORDS = set("""
a about after all also am an and any are as at be been before but by can could did do does for from had has have
he her here him his how i if in into is it its me my no not of on or our she so than that the their them then there
these they this those to up us was we were what when where which who whom why will with would you your
""".split())


def tokenize(text):
    return [token.lower() for token in TOKEN.findall(text)]
