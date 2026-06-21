from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results" / "word_cat"
POS_FIELD = "pos_tag"
GROUP_FIELD = "pos_group"
GROUP_ORDER = ["open_class", "closed_class", "other"]
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


def iter_pos_count_files():
    yield from sorted(RESULTS_DIR.rglob("*_pos_counts_*.csv"))


def format_percentage(count: int, total: int) -> str:
    if total == 0:
        return "0.0000"
    return f"{count / total * 100:.4f}"


def read_aggregated_counts(input_file: Path) -> Counter[str]:
    counts: Counter[str] = Counter()

    with input_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {POS_FIELD, "count"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{input_file} is missing required column(s): {missing}")

        for row_number, row in enumerate(reader, start=2):
            pos_tag = row[POS_FIELD]
            if pos_tag == "TOTAL":
                continue

            try:
                count = int(row["count"])
            except ValueError as error:
                raise ValueError(
                    f"Invalid count at {input_file}:{row_number}"
                ) from error

            try:
                group = POS_GROUPS[pos_tag]
            except KeyError as error:
                raise ValueError(
                    f"No aggregate group configured for POS tag {pos_tag!r} "
                    f"in {input_file}"
                ) from error

            counts[group] += count

    return counts


def write_counts_csv(counts: Counter[str], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    total = sum(counts.values())

    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=[GROUP_FIELD, "count", "percentage"]
        )
        writer.writeheader()

        for group in GROUP_ORDER:
            count = counts.get(group, 0)
            writer.writerow(
                {
                    GROUP_FIELD: group,
                    "count": count,
                    "percentage": format_percentage(count, total),
                }
            )

        writer.writerow(
            {
                GROUP_FIELD: "TOTAL",
                "count": total,
                "percentage": "100.0000",
            }
        )


def write_distribution_plot(counts: Counter[str], output_file: Path, title: str) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    values = [counts.get(group, 0) for group in GROUP_ORDER]

    plt.figure(figsize=(8, 6))
    plt.bar(GROUP_ORDER, values)
    plt.xlabel("Aggregated word category")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()


def output_stem(input_file: Path) -> str:
    return input_file.stem.replace("_pos_counts_", "_pos_group_counts_")


def plot_title(input_file: Path) -> str:
    parts = input_file.stem.split("_")
    if len(parts) < 5:
        return f"Distribution of aggregated word categories for {input_file.stem}"

    source = "reference" if parts[0] == "ref" else "hypothesis"
    speech_type = "HMI speech" if parts[-2] == "hmi" else "read speech"
    asr_system = "Chirp" if parts[-1] == "google" else "Whisper"
    return (
        f"Distribution of {source} aggregated word categories "
        f"for {speech_type} ({asr_system})"
    )


def main() -> None:
    input_files = list(iter_pos_count_files())
    if not input_files:
        raise FileNotFoundError(f"No POS count CSV files found under {RESULTS_DIR}")

    for input_file in input_files:
        counts = read_aggregated_counts(input_file)
        stem = output_stem(input_file)
        csv_output = input_file.with_name(f"{stem}.csv")
        plot_output = input_file.with_name(f"{stem}_distribution.png")

        write_counts_csv(counts, csv_output)
        write_distribution_plot(counts, plot_output, plot_title(input_file))

        print(f"Wrote aggregated POS group counts to {csv_output}")
        print(f"Wrote aggregated POS group plot to {plot_output}")


if __name__ == "__main__":
    main()
