"""Regenerate MIID/miner/homophones_en.json from the validator's English pool.

The voice word check (MIID/miner/word_check.py) compares what Whisper wrote
with the target words. Whisper spells a homophone the commoner way ("pair"
for a correctly spoken "pear", "brews" for "bruise"), which the check would
otherwise count as a wrong word although no listener could hear a
difference. This script lists, for every word in
MIID/validator/word_pools/en.txt and every English digit word, the other
spellings with the same pronunciation in the CMU Pronouncing Dictionary.

Stress marks are ignored, and a vowel followed by an unstressed "ER" is
treated like the vowel followed by "R" (CMU has "choir" as K W AY ER but
"quire" as K W AY R).

Run it again whenever upstream changes the English word pool:

    pip install --target /tmp/cmudict cmudict
    PYTHONPATH=/tmp/cmudict python scripts/miner/build_homophones.py
"""

import json
import re
import sys
from pathlib import Path

import cmudict

ROOT = Path(__file__).resolve().parents[2]
POOL = ROOT / "MIID" / "validator" / "word_pools" / "en.txt"
OUT = ROOT / "MIID" / "miner" / "homophones_en.json"
DIGITS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}


def sound(phones):
    plain = [re.sub(r"\d", "", p) for p in phones]
    out = []
    for p in plain:
        if p == "ER" and out and out[-1] in VOWELS:
            p = "R"
        out.append(p)
    return tuple(out)


def main() -> int:
    pronunciations = cmudict.dict()
    by_sound = {}
    for word, prons in pronunciations.items():
        if not re.fullmatch(r"[a-z]+", word):
            continue
        for phones in prons:
            by_sound.setdefault(sound(phones), set()).add(word)

    pool = []
    for line in POOL.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            pool.append(line.split()[0].lower())

    table = {}
    for word in sorted(set(pool + DIGITS)):
        same = set()
        for phones in pronunciations.get(word, []):
            same |= by_sound.get(sound(phones), set())
        same.discard(word)
        if same:
            table[word] = sorted(same)

    OUT.write_text(json.dumps(table, indent=0, sort_keys=True) + "\n", encoding="utf-8")
    unknown = sorted(w for w in set(pool) if w not in pronunciations)
    print(f"{len(table)} of {len(set(pool))} pool words have homophones -> {OUT}")
    if unknown:
        print(f"not in CMU (checked strictly): {', '.join(unknown)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
