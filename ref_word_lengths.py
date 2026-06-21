from __future__ import annotations

import csv
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
REF_FILE = SCRIPT_DIR / "data" / "google" / "nl_hmi_g4" / "score_wer" / "ref.trn"
OUTPUT_FILE = (
    SCRIPT_DIR / "results" / "word_length" / "hmi" / "ref_word_lengths_hmi_google.csv"
)
OUTPUT_PLOT_FILE = OUTPUT_FILE.with_name(f"{OUTPUT_FILE.stem}_distribution.png")
MAX_ROWS = None
PRINT_WORDS = False


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


def count_word_lengths(
    ref_file: Path, max_rows: int | None = None, print_words: bool = False
) -> Counter[int]:
    counts: Counter[int] = Counter()

    for tokens in parse_ref_tokens(ref_file, max_rows=max_rows):
        lengths = [len(token) for token in tokens]
        counts.update(lengths)

        if print_words:
            for token, word_length in zip(tokens, lengths):
                print(f"{token}\t{word_length}")

    return counts


def write_counts_csv(counts: Counter[int], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["word_length", "count"])
        writer.writeheader()

        for word_length, count in sorted(counts.items()):
            writer.writerow({"word_length": word_length, "count": count})

        writer.writerow({"word_length": "TOTAL", "count": sum(counts.values())})


def write_length_distribution_plot(
    counts: Counter[int], output_file: Path
) -> None:
    if not counts:
        raise ValueError("No word lengths to plot")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    lengths = sorted(counts)
    count_values = [counts[length] for length in lengths]

    plt.figure(figsize=(10, 6))
    plt.bar(lengths, count_values, width=0.8)
    plt.xlabel("Word length")
    plt.ylabel("Count")
    plt.title("Distribution of word lengths")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()


def main() -> None:
    total_start = time.perf_counter()

    if not REF_FILE.exists():
        raise FileNotFoundError(f"Could not find ref.trn file: {REF_FILE}")

    processing_start = time.perf_counter()
    counts = count_word_lengths(
        REF_FILE, max_rows=MAX_ROWS, print_words=PRINT_WORDS
    )
    processing_seconds = time.perf_counter() - processing_start

    write_counts_csv(counts, OUTPUT_FILE)
    write_length_distribution_plot(counts, OUTPUT_PLOT_FILE)
    total_seconds = time.perf_counter() - total_start

    print(f"Wrote {sum(counts.values())} reference word lengths to {OUTPUT_FILE}")
    print(f"Wrote word length distribution plot to {OUTPUT_PLOT_FILE}")
    print(f"Processing time: {processing_seconds:.3f} seconds")
    print(f"Total runtime: {total_seconds:.3f} seconds")


if __name__ == "__main__":
    main()
