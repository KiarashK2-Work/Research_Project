from __future__ import annotations

import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results" / "word_cat"
INPUT_FILE = RESULTS_DIR / "word_cat_errors_combined_all.csv"
OUTPUT_FILE = RESULTS_DIR / "word_cat_group_errors_combined_all.csv"
POS_FIELD = "pos_tag"
GROUP_FIELD = "pos_group"
GROUP_ORDER = ["open_class", "closed_class", "other"]
COUNT_FIELDS = [
    "deletion_source_count",
    "substitution_source_count",
    "substitution_destination_count",
    "insertion_destination_count",
]
POS_GROUPS = {
    "ADJ": "open_class",
    "ADV": "open_class",
    "INTJ": "open_class",
    "NOUN": "open_class",
    "PROPN": "open_class",
    "VERB": "open_class",
    "ADP": "closed_class",
    "AUX": "closed_class",
    "CCONJ": "closed_class",
    "DET": "closed_class",
    "NUM": "closed_class",
    "PART": "closed_class",
    "PRON": "closed_class",
    "SCONJ": "closed_class",
    "PUNCT": "other",
    "SYM": "other",
    "X": "other",
}


def empty_counts() -> dict[str, int]:
    return {field: 0 for field in COUNT_FIELDS}


def read_aggregated_rows(input_file: Path) -> list[dict[str, str | int]]:
    aggregated: dict[tuple[str, str, str], dict[str, int]] = {}

    with input_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"speech_type", "model", POS_FIELD, *COUNT_FIELDS}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{input_file} is missing required column(s): {missing}")

        for row_number, row in enumerate(reader, start=2):
            pos_tag = row[POS_FIELD]
            try:
                pos_group = POS_GROUPS[pos_tag]
            except KeyError as error:
                raise ValueError(
                    f"No aggregate group configured for POS tag {pos_tag!r} "
                    f"in {input_file}"
                ) from error

            key = (row["speech_type"], row["model"], pos_group)
            counts = aggregated.setdefault(key, empty_counts())

            for count_field in COUNT_FIELDS:
                try:
                    counts[count_field] += int(row[count_field])
                except ValueError as error:
                    raise ValueError(
                        f"Invalid {count_field} at {input_file}:{row_number}"
                    ) from error

    rows: list[dict[str, str | int]] = []
    for speech_type, model in sorted(
        {(key[0], key[1]) for key in aggregated},
        key=lambda item: (item[0], item[1]),
    ):
        for pos_group in GROUP_ORDER:
            key = (speech_type, model, pos_group)
            if key not in aggregated:
                continue

            row: dict[str, str | int] = {
                "speech_type": speech_type,
                "model": model,
                GROUP_FIELD: pos_group,
            }
            row.update(aggregated[key])
            rows.append(row)

    return rows


def write_csv(rows: list[dict[str, str | int]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["speech_type", "model", GROUP_FIELD, *COUNT_FIELDS]

    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Could not find combined error CSV: {INPUT_FILE}")

    rows = read_aggregated_rows(INPUT_FILE)
    write_csv(rows, OUTPUT_FILE)
    print(f"Wrote {len(rows)} aggregated word-category error rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
