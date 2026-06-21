from __future__ import annotations

import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results" / "word_cat"
OUTPUT_FILE = (
    RESULTS_DIR
    / "top10_substitution_pos_pairs_by_category_error_rate_min100.csv"
)
TOP_N = 10
MIN_SOURCE_CATEGORY_WORDS = 100

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
        "file_key": "google",
        "label": "chirp",
    },
    "whisper": {
        "result_dir": "whisper",
        "file_key": "whisper",
        "label": "whisper",
    },
}


def pair_file_for(speech_key: str, asr_config: dict[str, str]) -> Path:
    asr_file_key = asr_config["file_key"]
    return (
        RESULTS_DIR
        / SPEECH_TYPES[speech_key]["result_dir"]
        / asr_config["result_dir"]
        / f"word_cat_errors_{speech_key}_{asr_file_key}_substitution_pos_pairs.csv"
    )


def ref_pos_counts_file_for(speech_key: str, asr_config: dict[str, str]) -> Path:
    asr_file_key = asr_config["file_key"]
    return (
        RESULTS_DIR
        / SPEECH_TYPES[speech_key]["result_dir"]
        / asr_config["result_dir"]
        / f"ref_pos_counts_{speech_key}_{asr_file_key}.csv"
    )


def read_source_denominators(counts_file: Path) -> dict[str, int]:
    denominators: dict[str, int] = {}

    with counts_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"pos_tag", "count"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{counts_file} is missing required column(s): {missing}")

        for row_number, row in enumerate(reader, start=2):
            pos_tag = row["pos_tag"]
            if pos_tag == "TOTAL":
                continue

            try:
                denominators[pos_tag] = int(row["count"])
            except ValueError as error:
                raise ValueError(
                    f"Invalid count at {counts_file}:{row_number}"
                ) from error

    return denominators


def read_pair_rows(pair_file: Path) -> list[dict[str, str]]:
    with pair_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {
            "source_pos",
            "destination_pos",
            "pair",
            "count",
            "percentage",
            "source_category_error_rate",
        }
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{pair_file} is missing required column(s): {missing}")

        return list(reader)


def top_rows_by_rate(
    rows: list[dict[str, str]],
    source_denominators: dict[str, int],
) -> list[dict[str, str]]:
    rows = [
        row
        for row in rows
        if source_denominators.get(row["source_pos"], 0) >= MIN_SOURCE_CATEGORY_WORDS
    ]
    return sorted(
        rows,
        key=lambda row: (
            -float(row["source_category_error_rate"] or 0),
            -int(row["count"]),
            row["source_pos"],
            row["destination_pos"],
        ),
    )[:TOP_N]


def build_output_rows() -> list[dict[str, str | int]]:
    output_rows: list[dict[str, str | int]] = []

    for speech_key, speech_config in SPEECH_TYPES.items():
        for _, asr_config in ASR_SYSTEMS.items():
            pair_file = pair_file_for(speech_key, asr_config)
            counts_file = ref_pos_counts_file_for(speech_key, asr_config)
            if not pair_file.exists():
                raise FileNotFoundError(f"Could not find POS pair CSV: {pair_file}")
            if not counts_file.exists():
                raise FileNotFoundError(
                    f"Could not find reference POS counts CSV: {counts_file}"
                )

            source_denominators = read_source_denominators(counts_file)
            top_rows = top_rows_by_rate(read_pair_rows(pair_file), source_denominators)
            for rank, row in enumerate(top_rows, start=1):
                output_rows.append(
                    {
                        "speech_type": speech_config["label"],
                        "model": asr_config["label"],
                        "rank": rank,
                        "source_pos": row["source_pos"],
                        "destination_pos": row["destination_pos"],
                        "pair": row["pair"],
                        "count": row["count"],
                        "percentage": row["percentage"],
                        "category_error_rate": row["source_category_error_rate"],
                        "source_category_reference_count": source_denominators.get(
                            row["source_pos"], 0
                        ),
                    }
                )

    return output_rows


def write_output_csv(rows: list[dict[str, str | int]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "speech_type",
        "model",
        "rank",
        "source_pos",
        "destination_pos",
        "pair",
        "count",
        "percentage",
        "category_error_rate",
        "source_category_reference_count",
    ]

    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = build_output_rows()
    write_output_csv(rows, OUTPUT_FILE)
    print(f"Wrote {len(rows)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
