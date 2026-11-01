# Usage (from the project root): python scripts/build_corpus.py

import os
import re
import textwrap
import urllib.request

SOURCE_URL = "https://www.gutenberg.org/cache/epub/1184/pg1184.txt"
OUTPUT_DIR = "corpus"
VOLUMES = ["ONE", "TWO", "THREE", "FOUR", "FIVE"]

# Facts the LLM cannot know from training: tells grounded answers from recalled ones.
PLANTED_FACTS = [
    {
        "id": "new-stationer",
        "kind": "new fact",
        "volume": 1,
        "action": "insert",
        "anchor": "“Bring them here.” The waiter did as he was desired.",
        "text": "The paper the waiter brought was of a pale blue colour; La Réserve "
                "bought it from a stationer named Mère Brigaud in the Rue de Rome, and "
                "every sheet bore in one corner a small watermark in the shape of a swallow.",
    },
    {
        "id": "split-seal-part1",
        "kind": "split fact (1/2)",
        "volume": 1,
        "action": "insert",
        "anchor": "And Danglars wrote the address as he spoke.",
        "text": "Before folding it, Danglars pressed into the corner of the sheet a "
                "small seal of green wax bearing the figure of an anchor, which he had "
                "taken that morning from the captain’s desk on board the Pharaon.",
    },
    {
        "id": "split-seal-part2",
        "kind": "split fact (2/2)",
        "volume": 2,
        "action": "insert",
        "anchor": "Emmanuel sighed, and summoned Cocles.",
        "text": "Every ship of the house of Morrel carried the same seal: an anchor cut "
                "in 1811 by Joseph Lanfrey, an engraver of Toulon, for old Morrel himself. "
                "There was no other seal of that design in all Marseilles.",
    },
    {
        "id": "new-ledger",
        "kind": "new fact",
        "volume": 4,
        "action": "insert",
        "anchor": "The count praised Bertuccio’s zeal, and ordered him to prepare for a",
        "text": "Bertuccio kept a ledger bound in brown calfskin, in which he wrote down "
                "every expense of the count since their arrival in Paris; its very first "
                "entry was four hundred and twelve francs paid to a glazier named "
                "Théophile Barrat for the windows of the house at Auteuil.",
    },
    {
        "id": "contradiction-arrival-date",
        "kind": "contradiction",
        "volume": 1,
        "action": "replace",
        "anchor": "On the 24th of February, 1815",
        "text": "On the 25th of February, 1815",
    },
    {
        "id": "contradiction-purse-ruby",
        "kind": "contradiction",
        "volume": 2,
        "action": "replace",
        "anchor": "diamond as large as a hazel-nut, with these words",
        "text": "ruby as large as a pigeon’s egg, with these words",
    },
    {
        "id": "contradiction-purse-ruby",
        "kind": "contradiction",
        "volume": 3,
        "action": "replace",
        "anchor": "the large diamond at the other",
        "text": "the large ruby at the other",
    },
]


def download(url):
    with urllib.request.urlopen(url) as response:
        text = response.read().decode("utf-8")
    return text.replace("\r\n", "\n")


def strip_gutenberg(text):
    start = text.index("*** START OF THE PROJECT GUTENBERG EBOOK")
    start = text.index("\n", start) + 1
    end = text.index("*** END OF THE PROJECT GUTENBERG EBOOK")
    body = text[start:end]
    body = body[:body.index("FOOTNOTES:")]
    # leftover illustration ids such as "20009m"
    return re.sub(r"^[ \t]*\d+m[ \t]*\n", "", body, flags=re.MULTILINE)


def split_volumes(body):
    # the first occurrence of each heading is in the table of contents
    starts = []
    for name in VOLUMES:
        matches = list(re.finditer(rf"^\s*VOLUME {name}\s*$", body, re.MULTILINE))
        starts.append(matches[-1].start())
    starts.append(len(body))
    return [body[starts[i]:starts[i + 1]].strip() + "\n" for i in range(len(VOLUMES))]


def plant_facts(volume_number, text):
    for fact in PLANTED_FACTS:
        if fact["volume"] != volume_number:
            continue
        count = text.count(fact["anchor"])
        if count != 1:
            raise ValueError(f"{fact['id']}: anchor found {count} times in volume {volume_number}, expected 1")
        if fact["action"] == "replace":
            text = text.replace(fact["anchor"], fact["text"])
        else:
            paragraph = textwrap.fill(fact["text"], width=70)
            end_of_paragraph = text.index("\n\n", text.index(fact["anchor"]))
            text = text[:end_of_paragraph] + "\n\n" + paragraph + text[end_of_paragraph:]
    return text


def report_planted_facts(volume_number, text, path):
    for fact in PLANTED_FACTS:
        if fact["volume"] != volume_number:
            continue
        planted = fact["text"] if fact["action"] == "replace" else textwrap.fill(fact["text"], width=70)
        start = text.index(planted)
        print(f"  planted {fact['id']} ({fact['kind']}): {path} chars {start}-{start + len(planted)}")


def main():
    body = strip_gutenberg(download(SOURCE_URL))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for number, volume in enumerate(split_volumes(body), start=1):
        volume = plant_facts(number, volume)
        path = os.path.join(OUTPUT_DIR, f"monte-cristo-vol{number}.txt")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(volume)
        print(f"{path}: {len(volume):,} characters")
        report_planted_facts(number, volume, path)


if __name__ == "__main__":
    main()
