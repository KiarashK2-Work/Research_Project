from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "results" / "word_cat" / "read" / "whisper"
DEFAULT_ERROR_FILE = INPUT_DIR / "word_cat_errors_read_whisper.csv"


def write_distribution_plot(
    counts: list[tuple[str, int]],
    output_file: Path,
    *,
    x_label: str,
    title: str,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    tags = [tag for tag, _ in counts]
    values = [count for _, count in counts]
    figure_width = max(10, len(tags) * 0.45)

    plt.figure(figsize=(figure_width, 6))
    plt.bar(tags, values)
    plt.xlabel(x_label)
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()


def write_counts_csv(
    counts: list[tuple[str, int]], output_file: Path, tag_column: str
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[tag_column, "count"])
        writer.writeheader()
        for tag, count in counts:
            writer.writerow({tag_column: tag, "count": count})


def count_error_pos_categories(
    input_file: Path, error_type: str, pos_column: str
) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()

    with input_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"error_type", pos_column}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{input_file} is missing required column(s): {missing}")

        for row in reader:
            if row["error_type"] != error_type:
                continue

            pos_tag = row[pos_column]
            if pos_tag:
                counts[pos_tag] += 1

    if not counts:
        raise ValueError(
            f"No {error_type} rows with {pos_column} values found in {input_file}"
        )

    return sorted(counts.items())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot word-category error distributions."
    )
    parser.add_argument(
        "--error-input",
        type=Path,
        default=DEFAULT_ERROR_FILE,
        help=f"Word-category error CSV. Defaults to {DEFAULT_ERROR_FILE}.",
    )
    parser.add_argument(
        "--deletion-pos-output",
        type=Path,
        help=(
            "Deletion source POS PNG. Defaults to "
            "<error-input stem>_deletion_source_pos.png."
        ),
    )
    parser.add_argument(
        "--deletion-pos-csv",
        type=Path,
        help=(
            "Deletion source POS CSV. Defaults to "
            "<error-input stem>_deletion_source_pos.csv."
        ),
    )
    parser.add_argument(
        "--substitution-source-pos-output",
        type=Path,
        help=(
            "Substitution source POS PNG. Defaults to "
            "<error-input stem>_substitution_source_pos.png."
        ),
    )
    parser.add_argument(
        "--substitution-source-pos-csv",
        type=Path,
        help=(
            "Substitution source POS CSV. Defaults to "
            "<error-input stem>_substitution_source_pos.csv."
        ),
    )
    parser.add_argument(
        "--substitution-destination-pos-output",
        type=Path,
        help=(
            "Substitution destination POS PNG. Defaults to "
            "<error-input stem>_substitution_destination_pos.png."
        ),
    )
    parser.add_argument(
        "--substitution-destination-pos-csv",
        type=Path,
        help=(
            "Substitution destination POS CSV. Defaults to "
            "<error-input stem>_substitution_destination_pos.csv."
        ),
    )
    parser.add_argument(
        "--insertion-pos-output",
        type=Path,
        help=(
            "Insertion destination POS PNG. Defaults to "
            "<error-input stem>_insertion_destination_pos.png."
        ),
    )
    parser.add_argument(
        "--insertion-pos-csv",
        type=Path,
        help=(
            "Insertion destination POS CSV. Defaults to "
            "<error-input stem>_insertion_destination_pos.csv."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    error_input = args.error_input.resolve()

    if not error_input.exists():
        raise FileNotFoundError(f"Could not find error input file: {error_input}")

    deletion_pos_output = args.deletion_pos_output or error_input.with_name(
        f"{error_input.stem}_deletion_source_pos.png"
    )
    substitution_source_pos_output = (
        args.substitution_source_pos_output
        or error_input.with_name(f"{error_input.stem}_substitution_source_pos.png")
    )
    substitution_destination_pos_output = (
        args.substitution_destination_pos_output
        or error_input.with_name(f"{error_input.stem}_substitution_destination_pos.png")
    )
    insertion_pos_output = args.insertion_pos_output or error_input.with_name(
        f"{error_input.stem}_insertion_destination_pos.png"
    )
    deletion_pos_csv = args.deletion_pos_csv or error_input.with_name(
        f"{error_input.stem}_deletion_source_pos.csv"
    )
    substitution_source_pos_csv = (
        args.substitution_source_pos_csv
        or error_input.with_name(f"{error_input.stem}_substitution_source_pos.csv")
    )
    substitution_destination_pos_csv = (
        args.substitution_destination_pos_csv
        or error_input.with_name(f"{error_input.stem}_substitution_destination_pos.csv")
    )
    insertion_pos_csv = args.insertion_pos_csv or error_input.with_name(
        f"{error_input.stem}_insertion_destination_pos.csv"
    )

    deletion_pos_counts = count_error_pos_categories(
        error_input, "deletion", "source_pos"
    )
    substitution_source_pos_counts = count_error_pos_categories(
        error_input, "substitution", "source_pos"
    )
    substitution_destination_pos_counts = count_error_pos_categories(
        error_input, "substitution", "destination_pos"
    )
    insertion_pos_counts = count_error_pos_categories(
        error_input, "insertion", "destination_pos"
    )

    write_distribution_plot(
        deletion_pos_counts,
        deletion_pos_output,
        x_label="Source POS tag",
        title="Deletions by source POS tag",
    )
    write_counts_csv(deletion_pos_counts, deletion_pos_csv, "source_pos")
    write_distribution_plot(
        substitution_source_pos_counts,
        substitution_source_pos_output,
        x_label="Source POS tag",
        title="Substitutions by source POS tag",
    )
    write_counts_csv(
        substitution_source_pos_counts,
        substitution_source_pos_csv,
        "source_pos",
    )
    write_distribution_plot(
        substitution_destination_pos_counts,
        substitution_destination_pos_output,
        x_label="Destination POS tag",
        title="Substitutions by destination POS tag",
    )
    write_counts_csv(
        substitution_destination_pos_counts,
        substitution_destination_pos_csv,
        "destination_pos",
    )
    write_distribution_plot(
        insertion_pos_counts,
        insertion_pos_output,
        x_label="Destination POS tag",
        title="Insertions by destination POS tag",
    )
    write_counts_csv(insertion_pos_counts, insertion_pos_csv, "destination_pos")

    print(f"Wrote deletion source POS plot to {deletion_pos_output}")
    print(f"Wrote deletion source POS table to {deletion_pos_csv}")
    print(f"Wrote substitution source POS plot to {substitution_source_pos_output}")
    print(f"Wrote substitution source POS table to {substitution_source_pos_csv}")
    print(
        "Wrote substitution destination POS plot to "
        f"{substitution_destination_pos_output}"
    )
    print(
        "Wrote substitution destination POS table to "
        f"{substitution_destination_pos_csv}"
    )
    print(f"Wrote insertion destination POS plot to {insertion_pos_output}")
    print(f"Wrote insertion destination POS table to {insertion_pos_csv}")


if __name__ == "__main__":
    main()
