from __future__ import annotations

import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results" / "word_cat"
OUTPUT_FILE = RESULTS_DIR / "word_cat_errors_combined_all.csv"

SPEECH_TYPES = {
    "read": {
        "result_dir": "read",
        "label": "read",
    },
    "hmi": {
        "result_dir": "hmi",
        "label": "hmi",
    },
}
ASR_SYSTEMS = {
    "chirp": {
        "result_dir": "google",
        "label": "chirp",
    },
    "whisper": {
        "result_dir": "whisper",
        "label": "whisper",
    },
}
ERROR_SERIES = {
    "deletion_source": {
        "filename_suffix": "deletion_source_pos.csv",
        "tag_column": "source_pos",
        "output_column": "deletion_source_count",
    },
    "substitution_source": {
        "filename_suffix": "substitution_source_pos.csv",
        "tag_column": "source_pos",
        "output_column": "substitution_source_count",
    },
    "substitution_destination": {
        "filename_suffix": "substitution_destination_pos.csv",
        "tag_column": "destination_pos",
        "output_column": "substitution_destination_count",
    },
    "insertion_destination": {
        "filename_suffix": "insertion_destination_pos.csv",
        "tag_column": "destination_pos",
        "output_column": "insertion_destination_count",
    },
}


def find_error_csv(input_dir: Path, filename_suffix: str) -> Path:
    matches = sorted(input_dir.glob(f"*_{filename_suffix}"))
    if not matches:
        raise FileNotFoundError(f"No *_{filename_suffix} file found in {input_dir}")
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise ValueError(
            f"Found multiple *_{filename_suffix} files in {input_dir}: {names}"
        )
    return matches[0]


def read_counts(input_file: Path, tag_column: str) -> dict[str, int]:
    counts: dict[str, int] = {}

    with input_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {tag_column, "count"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{input_file} is missing required column(s): {missing}")

        for row_number, row in enumerate(reader, start=2):
            tag = row[tag_column]
            if tag == "TOTAL":
                continue

            try:
                counts[tag] = int(row["count"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid count at {input_file}:{row_number}"
                ) from error

    return counts


def load_error_counts(input_dir: Path) -> dict[str, dict[str, int]]:
    combined_counts: dict[str, dict[str, int]] = {}

    for series_name, config in ERROR_SERIES.items():
        input_file = find_error_csv(input_dir, str(config["filename_suffix"]))
        combined_counts[series_name] = read_counts(input_file, str(config["tag_column"]))

    return combined_counts


def build_rows() -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []

    for speech_key, speech_config in SPEECH_TYPES.items():
        for asr_key, asr_config in ASR_SYSTEMS.items():
            input_dir = (
                RESULTS_DIR
                / speech_config["result_dir"]
                / asr_config["result_dir"]
            )
            combined_counts = load_error_counts(input_dir)
            tags = sorted(
                {
                    tag
                    for series_counts in combined_counts.values()
                    for tag in series_counts
                }
            )

            for tag in tags:
                row: dict[str, str | int] = {
                    "speech_type": speech_config["label"],
                    "model": asr_config["label"],
                    "pos_tag": tag,
                }
                for series_name, config in ERROR_SERIES.items():
                    row[str(config["output_column"])] = combined_counts[
                        series_name
                    ].get(tag, 0)
                rows.append(row)

    return rows


def write_combined_csv(rows: list[dict[str, str | int]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "speech_type",
        "model",
        "pos_tag",
        "deletion_source_count",
        "substitution_source_count",
        "substitution_destination_count",
        "insertion_destination_count",
    ]

    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = build_rows()
    write_combined_csv(rows, OUTPUT_FILE)
    print(f"Wrote {len(rows)} combined word-category error rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
