from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results" / "word_cat"
OUTPUT_FILE = RESULTS_DIR / "substitution_group_pairs_all.csv"
GROUP_ORDER = ["open_class", "closed_class", "other"]

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


def pair_file_for(speech_key: str, asr_config: dict[str, str]) -> Path:
    asr_file_key = asr_config["file_key"]
    return (
        RESULTS_DIR
        / SPEECH_TYPES[speech_key]["result_dir"]
        / asr_config["result_dir"]
        / f"word_cat_errors_{speech_key}_{asr_file_key}_substitution_pos_pairs.csv"
    )


def ref_group_counts_file_for(speech_key: str, asr_config: dict[str, str]) -> Path:
    asr_file_key = asr_config["file_key"]
    return (
        RESULTS_DIR
        / SPEECH_TYPES[speech_key]["result_dir"]
        / asr_config["result_dir"]
        / f"ref_pos_group_counts_{speech_key}_{asr_file_key}.csv"
    )


def format_percentage(count: int, total: int) -> str:
    if total == 0:
        return "0.0000"
    return f"{count / total * 100:.4f}"


def format_rate(count: int, denominator: int) -> str:
    if denominator == 0:
        return ""
    return f"{count / denominator:.6f}"


def read_ref_group_denominators(counts_file: Path) -> dict[str, int]:
    denominators: dict[str, int] = {}

    with counts_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"pos_group", "count"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{counts_file} is missing required column(s): {missing}")

        for row_number, row in enumerate(reader, start=2):
            pos_group = row["pos_group"]
            if pos_group == "TOTAL":
                continue

            try:
                denominators[pos_group] = int(row["count"])
            except ValueError as error:
                raise ValueError(
                    f"Invalid count at {counts_file}:{row_number}"
                ) from error

    return denominators


def read_group_pair_counts(pair_file: Path) -> tuple[Counter[tuple[str, str]], int]:
    pair_counts: Counter[tuple[str, str]] = Counter()
    total_substitutions = 0

    with pair_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"source_pos", "destination_pos", "count"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{pair_file} is missing required column(s): {missing}")

        for row_number, row in enumerate(reader, start=2):
            source_pos = row["source_pos"]
            destination_pos = row["destination_pos"]

            try:
                source_group = POS_GROUPS[source_pos]
                destination_group = POS_GROUPS[destination_pos]
            except KeyError as error:
                raise ValueError(
                    f"No aggregate group configured for POS pair "
                    f"{source_pos!r} -> {destination_pos!r} in {pair_file}"
                ) from error

            try:
                count = int(row["count"])
            except ValueError as error:
                raise ValueError(
                    f"Invalid count at {pair_file}:{row_number}"
                ) from error

            pair_counts[(source_group, destination_group)] += count
            total_substitutions += count

    return pair_counts, total_substitutions


def build_rows() -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []

    for speech_key, speech_config in SPEECH_TYPES.items():
        for _, asr_config in ASR_SYSTEMS.items():
            pair_file = pair_file_for(speech_key, asr_config)
            denominators_file = ref_group_counts_file_for(speech_key, asr_config)
            if not pair_file.exists():
                raise FileNotFoundError(f"Could not find POS pair CSV: {pair_file}")
            if not denominators_file.exists():
                raise FileNotFoundError(
                    f"Could not find reference group counts CSV: {denominators_file}. "
                    "Run aggregate_pos_group_distributions.py first."
                )

            pair_counts, total_substitutions = read_group_pair_counts(pair_file)
            source_denominators = read_ref_group_denominators(denominators_file)

            for source_group in GROUP_ORDER:
                for destination_group in GROUP_ORDER:
                    count = pair_counts.get((source_group, destination_group), 0)
                    rows.append(
                        {
                            "speech_type": speech_config["label"],
                            "model": asr_config["label"],
                            "source_group": source_group,
                            "destination_group": destination_group,
                            "pair": f"{source_group} -> {destination_group}",
                            "count": count,
                            "percentage": format_percentage(
                                count, total_substitutions
                            ),
                            "source_group_error_rate": format_rate(
                                count, source_denominators.get(source_group, 0)
                            ),
                            "total_substitutions": total_substitutions,
                            "source_group_reference_count": source_denominators.get(
                                source_group, 0
                            ),
                        }
                    )

    return rows


def write_csv(rows: list[dict[str, str | int]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "speech_type",
        "model",
        "source_group",
        "destination_group",
        "pair",
        "count",
        "percentage",
        "source_group_error_rate",
        "total_substitutions",
        "source_group_reference_count",
    ]

    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = build_rows()
    write_csv(rows, OUTPUT_FILE)
    print(f"Wrote {len(rows)} substitution group-pair rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
