from __future__ import annotations

import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILE = (
    SCRIPT_DIR
    / "results"
    / "word_cat"
    / "hmi"
    / "word_cat_errors_hmi_google.csv"
)
OUTPUT_FILE = (
    SCRIPT_DIR
    / "results"
    / "word_length"
    / "hmi"
    / "word_length_errors_hmi_google.csv"
)


def word_length(word: str) -> int | str:
    if word in {"<ins>", "<del>"}:
        return word
    return len(word)


def parse_word_length_error_rows(input_file: Path) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []

    with input_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {
            "utterance_id",
            "token_index",
            "error_type",
            "source_word",
            "destination_word",
        }
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{input_file} is missing required column(s): {missing}")

        for row in reader:
            source_word = row["source_word"]
            destination_word = row["destination_word"]
            rows.append(
                {
                    "utterance_id": row["utterance_id"],
                    "token_index": row["token_index"],
                    "error_type": row["error_type"],
                    "source_word": source_word,
                    "source_word_length": word_length(source_word),
                    "destination_word": destination_word,
                    "destination_word_length": word_length(destination_word),
                }
            )

    return rows


def write_csv(rows: list[dict[str, str | int]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "utterance_id",
        "token_index",
        "error_type",
        "source_word",
        "source_word_length",
        "destination_word",
        "destination_word_length",
    ]

    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Could not find word-category error file: {INPUT_FILE}")

    rows = parse_word_length_error_rows(INPUT_FILE)
    write_csv(rows, OUTPUT_FILE)
    print(f"Wrote {len(rows)} word-length errors to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
