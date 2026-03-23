"""
Download Project Gutenberg EPUB files.
---------------------------------------
Fetches a batch of public domain EPUBs from Project Gutenberg's
robot-accessible mirrors.

Usage:
    python scripts/download_gutenberg_epubs.py [--count 100] [--output data/raw_downloads/rare_formats/gutenberg_epubs]
"""

import os
import sys
import time
import urllib.request
from pathlib import Path

# Gutenberg EPUB mirror pattern
# Books are at: https://www.gutenberg.org/ebooks/{id}.epub.images
# or: https://www.gutenberg.org/cache/epub/{id}/pg{id}.epub
GUTENBERG_CACHE = "https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.epub"

# Well-known public domain books with reliable EPUB availability
SEED_BOOK_IDS = [
    1342,   # Pride and Prejudice
    11,     # Alice's Adventures in Wonderland
    1661,   # Sherlock Holmes
    84,     # Frankenstein
    98,     # A Tale of Two Cities
    1232,   # The Prince (Machiavelli)
    74,     # Tom Sawyer
    2701,   # Moby Dick
    1952,   # The Yellow Wallpaper
    345,    # Dracula
    1080,   # A Modest Proposal
    76,     # Huckleberry Finn
    2591,   # Grimm's Fairy Tales
    16328,  # Beowulf
    4300,   # Ulysses
    1400,   # Great Expectations
    844,    # The Importance of Being Earnest
    2554,   # Crime and Punishment
    1260,   # Jane Eyre
    174,    # Picture of Dorian Gray
    36,     # War of the Worlds
    55,     # Wonderful Wizard of Oz
    1184,   # Count of Monte Cristo
    25344,  # Scarlet Letter
    2814,   # Dubliners
    120,    # Treasure Island
    3207,   # Leviathan
    135,    # Les Miserables
    5200,   # Metamorphosis
    43,     # Jekyll and Hyde
    996,    # Don Quixote
    46,     # A Christmas Carol
    1497,   # Republic (Plato)
    1727,   # The Odyssey
    6130,   # The Iliad
    514,    # Little Women
    768,    # Wuthering Heights
    161,    # Sense and Sensibility
    209,    # Turn of the Screw
    1998,   # Thus Spake Zarathustra
    3600,   # Essays of Montaigne
    779,    # Divine Comedy (Dante)
    2680,   # Meditations (Marcus Aurelius)
    7370,   # Second Treatise of Government
    4363,   # Beyond Good and Evil
    1250,   # Anthem (Ayn Rand)
    35,     # Time Machine
    219,    # Heart of Darkness
    408,    # Souls of Black Folk
    1322,   # Leaves of Grass
]


def download_epub(book_id: int, output_dir: Path) -> bool:
    """Download a single EPUB from Gutenberg. Returns True on success."""
    url = GUTENBERG_CACHE.format(book_id=book_id)
    dest = output_dir / f"pg{book_id}.epub"
    if dest.exists() and dest.stat().st_size > 1000:
        return True  # Already downloaded

    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "JCoder-Research/1.0 (AI research; contact: none)",
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
            if len(data) < 500:
                return False
            dest.write_bytes(data)
            return True
    except Exception as exc:
        print(f"  [FAIL] Book {book_id}: {exc}")
        return False


def main():
    count = 50
    output = Path("data/raw_downloads/rare_formats/gutenberg_epubs")

    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--count" and i + 1 < len(sys.argv) - 1:
            count = int(sys.argv[i + 2])
        elif arg == "--output" and i + 1 < len(sys.argv) - 1:
            output = Path(sys.argv[i + 2])

    output.mkdir(parents=True, exist_ok=True)
    book_ids = SEED_BOOK_IDS[:count]

    print(f"Downloading {len(book_ids)} EPUBs to {output}")
    success = 0
    for book_id in book_ids:
        if download_epub(book_id, output):
            success += 1
            print(f"  [OK] pg{book_id}.epub")
        time.sleep(0.5)  # Be polite to Gutenberg

    print(f"\nDone: {success}/{len(book_ids)} EPUBs downloaded to {output}")


if __name__ == "__main__":
    os.chdir(os.environ.get("JCODER_ROOT", str(Path(__file__).resolve().parent.parent)))
    main()
