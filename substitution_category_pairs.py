from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results" / "word_cat"

SPEECH_TYPES = {
    "read": {
        "result_dir": "read",
    },
    "hmi": {
        "result_dir": "hmi",
    },
}
ASR_SYSTEMS = {
    "google": {
        "result_dir": "google",
    },
    "whisper": {
        "result_dir": "whisper",
    },
}


def input_file_for(speech_key: str, asr_key: str) -> Path:
    return (
        RESULTS_DIR
        / SPEECH_TYPES[speech_key]["result_dir"]
        / ASR_SYSTEMS[asr_key]["result_dir"]
        / f"word_cat_errors_{speech_key}_{asr_key}.csv"
    )


def output_file_for(input_file: Path) -> Path:
    return input_file.with_name(
        input_file.name.replace(".csv", "_substitution_pos_pairs.csv")
    )


def ref_pos_counts_file_for(input_file: Path) -> Path:
    matches = sorted(input_file.parent.glob("ref_pos_counts_*.csv"))
    if not matches:
        raise FileNotFoundError(f"Could not find ref_pos_counts CSV in {input_file.parent}")
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise ValueError(
            f"Found multiple ref_pos_counts CSVs in {input_file.parent}: {names}"
        )
    return matches[0]


def format_percentage(count: int, total: int) -> str:
    if total == 0:
        return "0.0000"
    return f"{count / total * 100:.4f}"


def format_rate(count: int, total: int) -> str:
    if total == 0:
        return ""
    return f"{count / total:.6f}"


def read_source_denominators(counts_file: Path) -> dict[str, int]:
    denominators: dict[str, int] = {}

    with counts_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"pos_tag", "count"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{counts_file} is missing required column(s): {missing}")

        for row in reader:
            pos_tag = row["pos_tag"]
            if pos_tag == "TOTAL":
                continue
            denominators[pos_tag] = int(row["count"])

    return denominators


def count_substitution_pairs(input_file: Path) -> tuple[Counter[tuple[str, str]], int]:
    pair_counts: Counter[tuple[str, str]] = Counter()
    total_substitutions = 0

    with input_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"error_type", "source_pos", "destination_pos"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{input_file} is missing required column(s): {missing}")

        for row in reader:
            if row["error_type"] != "substitution":
                continue

            source_pos = row["source_pos"]
            destination_pos = row["destination_pos"]
            pair_counts[(source_pos, destination_pos)] += 1
            total_substitutions += 1

    return pair_counts, total_substitutions


def write_pair_counts_csv(
    pair_counts: Counter[tuple[str, str]],
    total_substitutions: int,
    source_denominators: dict[str, int],
    output_file: Path,
) -> None:
    rows = [
        {
            "source_pos": source_pos,
            "destination_pos": destination_pos,
            "pair": f"{source_pos} -> {destination_pos}",
            "count": count,
            "percentage": format_percentage(count, total_substitutions),
            "source_category_error_rate": format_rate(
                count, source_denominators.get(source_pos, 0)
            ),
        }
        for (source_pos, destination_pos), count in pair_counts.items()
    ]
    rows.sort(
        key=lambda row: (
            -int(row["count"]),
            str(row["source_pos"]),
            str(row["destination_pos"]),
        )
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_pos",
                "destination_pos",
                "pair",
                "count",
                "percentage",
                "source_category_error_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    for speech_key in SPEECH_TYPES:
        for asr_key in ASR_SYSTEMS:
            input_file = input_file_for(speech_key, asr_key)
            if not input_file.exists():
                raise FileNotFoundError(f"Could not find word category file: {input_file}")

            pair_counts, total_substitutions = count_substitution_pairs(input_file)
            source_denominators = read_source_denominators(
                ref_pos_counts_file_for(input_file)
            )
            output_file = output_file_for(input_file)
            write_pair_counts_csv(
                pair_counts,
                total_substitutions,
                source_denominators,
                output_file,
            )
            print(
                f"Wrote {len(pair_counts)} substitution POS pairs "
                f"from {total_substitutions} substitutions to {output_file}"
            )


if __name__ == "__main__":
    main()
