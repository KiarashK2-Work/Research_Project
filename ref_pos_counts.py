from __future__ import annotations

import csv
import argparse
import time
from collections import Counter
from pathlib import Path

import spacy
from spacy.tokens import Doc


SCRIPT_DIR = Path(__file__).resolve().parent
REF_FILE = SCRIPT_DIR / "data" / "google" / "nl_read_non-native" / "score_wer" / "ref.trn"
OUTPUT_FILE = SCRIPT_DIR / "ref_pos_counts.csv"
SPACY_MODEL = "nl_core_news_md"


def parse_ref_tokens(ref_file: Path, max_rows: int | None = None) -> list[list[str]]:
    utterances: list[list[str]] = []

    with ref_file.open(encoding="utf-8", errors="replace") as handle:
        for row_index, line in enumerate(handle):
            if max_rows is not None and row_index >= max_rows:
                break
            text = line.rsplit("\t", 1)[0].strip()
            if text:
                utterances.append(text.split())

    return utterances


def tag_tokens(nlp, tokens: list[str]) -> list[str]:
    doc = Doc(nlp.vocab, words=[token.lower() for token in tokens])
    for _, component in nlp.pipeline:
        doc = component(doc)
    return [token.pos_ for token in doc]


def count_pos_tags(
    ref_file: Path, nlp, max_rows: int | None = None, print_words: bool = False
) -> Counter[str]:
    counts: Counter[str] = Counter()

    for tokens in parse_ref_tokens(ref_file, max_rows=max_rows):
        pos_tags = tag_tokens(nlp, tokens)
        counts.update(pos_tags)

        if print_words:
            for token, pos_tag in zip(tokens, pos_tags):
                print(f"{token}\t{pos_tag}")

    return counts


def write_counts_csv(counts: Counter[str], output_file: Path) -> None:
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pos_tag", "count"])
        writer.writeheader()

        for pos_tag, count in sorted(counts.items()):
            writer.writerow({"pos_tag": pos_tag, "count": count})

        writer.writerow({"pos_tag": "TOTAL", "count": sum(counts.values())})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count POS tags in ref.trn.")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Only process the first N rows from ref.trn.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help=f"CSV output path. Defaults to {OUTPUT_FILE.name}.",
    )
    parser.add_argument(
        "--print-words",
        action="store_true",
        help="Print each word and its POS tag while processing.",
    )
    return parser.parse_args()


def main() -> None:
    total_start = time.perf_counter()
    args = parse_args()

    if not REF_FILE.exists():
        raise FileNotFoundError(f"Could not find ref.trn file: {REF_FILE}")

    try:
        nlp = spacy.load(SPACY_MODEL)
    except OSError as exc:
        raise SystemExit(
            f"Could not load spaCy model '{SPACY_MODEL}'. Install it with:\n"
            f"python -m spacy download {SPACY_MODEL}"
        ) from exc

    processing_start = time.perf_counter()
    counts = count_pos_tags(
        REF_FILE, nlp, max_rows=args.max_rows, print_words=args.print_words
    )
    processing_seconds = time.perf_counter() - processing_start

    write_counts_csv(counts, args.output)
    total_seconds = time.perf_counter() - total_start

    print(f"Wrote {sum(counts.values())} POS-tagged reference words to {args.output}")
    print(f"Processing time: {processing_seconds:.3f} seconds")
    print(f"Total runtime: {total_seconds:.3f} seconds")


if __name__ == "__main__":
    main()
