# build_corpus.py downloads The Count of Monte Cristo from Project Gutenberg
# and splits it into one .txt file per volume in corpus/.
#
# Run from the project root:  python scripts/build_corpus.py

import os
import re
import urllib.request

SOURCE_URL = "https://www.gutenberg.org/cache/epub/1184/pg1184.txt"
OUTPUT_DIR = "corpus"
VOLUMES = ["ONE", "TWO", "THREE", "FOUR", "FIVE"]


def download(url):
    with urllib.request.urlopen(url) as response:
        text = response.read().decode("utf-8")
    # Gutenberg serves CRLF line endings; normalise so offsets are the same on every OS
    return text.replace("\r\n", "\n")


def strip_gutenberg(text):
    # keep only the book: drop the licence header and footer
    start = text.index("*** START OF THE PROJECT GUTENBERG EBOOK")
    start = text.index("\n", start) + 1
    end = text.index("*** END OF THE PROJECT GUTENBERG EBOOK")
    body = text[start:end]
    # the editor's footnotes sit after the last chapter and belong to every volume,
    # so they are dropped rather than attached to volume five
    body = body[:body.index("FOOTNOTES:")]
    # lines like "20009m" are illustration ids left over from the Gutenberg HTML edition
    # ([ \t] rather than \s, so the blank lines that separate paragraphs are kept)
    return re.sub(r"^[ \t]*\d+m[ \t]*\n", "", body, flags=re.MULTILINE)


def split_volumes(body):
    # each "VOLUME X" heading appears twice: once in the table of contents, once in the text.
    # The last occurrence is the real one.
    starts = []
    for name in VOLUMES:
        matches = list(re.finditer(rf"^\s*VOLUME {name}\s*$", body, re.MULTILINE))
        starts.append(matches[-1].start())
    starts.append(len(body))
    return [body[starts[i]:starts[i + 1]].strip() + "\n" for i in range(len(VOLUMES))]


def main():
    body = strip_gutenberg(download(SOURCE_URL))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for number, volume in enumerate(split_volumes(body), start=1):
        path = os.path.join(OUTPUT_DIR, f"monte-cristo-vol{number}.txt")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(volume)
        print(f"{path}: {len(volume):,} characters")


if __name__ == "__main__":
    main()
