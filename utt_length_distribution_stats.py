from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_ROOT = SCRIPT_DIR / "results" / "utt_length"
OUTPUT_FILE = INPUT_ROOT / "utt_length_distribution_stats.csv"
SPEECH_TYPES = ("read", "hmi")
MODELS = ("google", "whisper")
CUTOFFS = (1, 2, 3, 4, 5, 6, 8)


def percentile(sorted_values: list[int], percentile_value: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot calculate percentile for empty values")

    index = (len(sorted_values) - 1) * percentile_value
    lower_index = math.floor(index)
    upper_index = math.ceil(index)
    if lower_index == upper_index:
        return float(sorted_values[lower_index])

    lower_weight = upper_index - index
    upper_weight = index - lower_index
    return (
        sorted_values[lower_index] * lower_weight
        + sorted_values[upper_index] * upper_weight
    )


def read_utterance_lengths(input_file: Path) -> list[int]:
    lengths: list[int] = []

    with input_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"utterance_length"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{input_file} is missing required column(s): {missing}")

        for row_number, row in enumerate(reader, start=2):
            try:
                utterance_length = int(row["utterance_length"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid utterance_length at {input_file}:{row_number}"
                ) from error

            if utterance_length <= 0:
                raise ValueError(
                    f"Invalid utterance_length at {input_file}:{row_number}: "
                    f"{utterance_length}"
                )

            lengths.append(utterance_length)

    if not lengths:
        raise ValueError(f"No utterance lengths found in {input_file}")

    return lengths


def summarize_lengths(
    lengths: list[int], speech: str, model: str
) -> dict[str, int | str | float]:
    sorted_lengths = sorted(lengths)
    count = len(sorted_lengths)
    mean = sum(sorted_lengths) / count
    variance = (
        sum((length - mean) ** 2 for length in sorted_lengths) / (count - 1)
        if count > 1
        else 0.0
    )
    counts = Counter(sorted_lengths)
    mode_count = max(counts.values())
    modes = [length for length, value in sorted(counts.items()) if value == mode_count]

    row: dict[str, int | str | float] = {
        "speech": speech,
        "model": model,
        "utterance_count": count,
        "min": sorted_lengths[0],
        "max": sorted_lengths[-1],
        "mean": round(mean, 6),
        "sd": round(math.sqrt(variance), 6),
        "variance": round(variance, 6),
        "median": round(percentile(sorted_lengths, 0.5), 6),
        "q1": round(percentile(sorted_lengths, 0.25), 6),
        "q3": round(percentile(sorted_lengths, 0.75), 6),
        "iqr": round(
            percentile(sorted_lengths, 0.75) - percentile(sorted_lengths, 0.25),
            6,
        ),
        "mode": ";".join(str(mode) for mode in modes),
        "mode_count": mode_count,
    }

    for cutoff in CUTOFFS:
        row[f"pct_le_{cutoff}"] = round(
            sum(length <= cutoff for length in sorted_lengths) / count,
            6,
        )
    row["pct_ge_7"] = round(sum(length >= 7 for length in sorted_lengths) / count, 6)

    return row


def write_stats_csv(
    rows: list[dict[str, int | str | float]], output_file: Path
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "speech",
        "model",
        "utterance_count",
        "min",
        "max",
        "mean",
        "sd",
        "variance",
        "median",
        "q1",
        "q3",
        "iqr",
        "mode",
        "mode_count",
        *(f"pct_le_{cutoff}" for cutoff in CUTOFFS),
        "pct_ge_7",
    ]

    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_stats(rows: list[dict[str, int | str | float]]) -> None:
    for row in rows:
        print(
            f"{row['speech']} {row['model']}: "
            f"n={row['utterance_count']}, "
            f"range={row['min']}-{row['max']}, "
            f"mean={row['mean']}, "
            f"sd={row['sd']}, "
            f"variance={row['variance']}, "
            f"median={row['median']}, "
            f"IQR={row['q1']}-{row['q3']}, "
            f"mode={row['mode']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate descriptive statistics for utterance length distributions."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=INPUT_ROOT,
        help=f"Root directory containing results/utt_length CSVs. Defaults to {INPUT_ROOT}.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=OUTPUT_FILE,
        help=f"Output summary CSV. Defaults to {OUTPUT_FILE}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()

    rows: list[dict[str, int | str | float]] = []
    for speech in SPEECH_TYPES:
        for model in MODELS:
            input_file = input_root / speech / model / f"utt_length_{speech}_{model}.csv"
            if not input_file.exists():
                raise FileNotFoundError(f"Could not find input file: {input_file}")

            lengths = read_utterance_lengths(input_file)
            rows.append(summarize_lengths(lengths, speech, model))

    output_csv = args.output_csv.resolve()
    write_stats_csv(rows, output_csv)
    print_stats(rows)
    print(f"Wrote utterance length distribution stats to {output_csv}")


if __name__ == "__main__":
    main()
