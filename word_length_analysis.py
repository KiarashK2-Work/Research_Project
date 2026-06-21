from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "results" / "word_length" / "hmi"
DEFAULT_ERROR_FILE = INPUT_DIR / "word_length_errors_hmi_google.csv"


def count_error_length_categories(
    input_file: Path, error_type: str, length_column: str
) -> list[tuple[int, int]]:
    counts: Counter[int] = Counter()

    with input_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"error_type", length_column}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{input_file} is missing required column(s): {missing}")

        for row_number, row in enumerate(reader, start=2):
            if row["error_type"] != error_type:
                continue

            length_value = row[length_column]
            if length_value in {"", "<ins>", "<del>"}:
                continue

            try:
                word_length = int(length_value)
            except ValueError as error:
                raise ValueError(
                    f"Invalid word length at {input_file}:{row_number}"
                ) from error

            counts[word_length] += 1

    if not counts:
        raise ValueError(
            f"No {error_type} rows with {length_column} values found in {input_file}"
        )

    return sorted(counts.items())


def write_distribution_plot(
    counts: list[tuple[int, int]],
    output_file: Path,
    *,
    x_label: str,
    title: str,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    lengths = [length for length, _ in counts]
    values = [count for _, count in counts]

    plt.figure(figsize=(10, 6))
    plt.bar(lengths, values, width=0.8)
    plt.xlabel(x_label)
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()


def write_counts_csv(
    counts: list[tuple[int, int]], output_file: Path, length_column: str
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[length_column, "count"])
        writer.writeheader()
        for word_length, count in counts:
            writer.writerow({length_column: word_length, "count": count})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot word-length error distributions."
    )
    parser.add_argument(
        "--error-input",
        type=Path,
        default=DEFAULT_ERROR_FILE,
        help=f"Word-length error CSV. Defaults to {DEFAULT_ERROR_FILE}.",
    )
    parser.add_argument(
        "--deletion-length-output",
        type=Path,
        help=(
            "Deletion source length PNG. Defaults to "
            "<error-input stem>_deletion_source_length.png."
        ),
    )
    parser.add_argument(
        "--deletion-length-csv",
        type=Path,
        help=(
            "Deletion source length CSV. Defaults to "
            "<error-input stem>_deletion_source_length.csv."
        ),
    )
    parser.add_argument(
        "--substitution-source-length-output",
        type=Path,
        help=(
            "Substitution source length PNG. Defaults to "
            "<error-input stem>_substitution_source_length.png."
        ),
    )
    parser.add_argument(
        "--substitution-source-length-csv",
        type=Path,
        help=(
            "Substitution source length CSV. Defaults to "
            "<error-input stem>_substitution_source_length.csv."
        ),
    )
    parser.add_argument(
        "--substitution-destination-length-output",
        type=Path,
        help=(
            "Substitution destination length PNG. Defaults to "
            "<error-input stem>_substitution_destination_length.png."
        ),
    )
    parser.add_argument(
        "--substitution-destination-length-csv",
        type=Path,
        help=(
            "Substitution destination length CSV. Defaults to "
            "<error-input stem>_substitution_destination_length.csv."
        ),
    )
    parser.add_argument(
        "--insertion-length-output",
        type=Path,
        help=(
            "Insertion destination length PNG. Defaults to "
            "<error-input stem>_insertion_destination_length.png."
        ),
    )
    parser.add_argument(
        "--insertion-length-csv",
        type=Path,
        help=(
            "Insertion destination length CSV. Defaults to "
            "<error-input stem>_insertion_destination_length.csv."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    error_input = args.error_input.resolve()

    if not error_input.exists():
        raise FileNotFoundError(f"Could not find error input file: {error_input}")

    deletion_length_output = args.deletion_length_output or error_input.with_name(
        f"{error_input.stem}_deletion_source_length.png"
    )
    deletion_length_csv = args.deletion_length_csv or error_input.with_name(
        f"{error_input.stem}_deletion_source_length.csv"
    )
    substitution_source_length_output = (
        args.substitution_source_length_output
        or error_input.with_name(
            f"{error_input.stem}_substitution_source_length.png"
        )
    )
    substitution_source_length_csv = (
        args.substitution_source_length_csv
        or error_input.with_name(
            f"{error_input.stem}_substitution_source_length.csv"
        )
    )
    substitution_destination_length_output = (
        args.substitution_destination_length_output
        or error_input.with_name(
            f"{error_input.stem}_substitution_destination_length.png"
        )
    )
    substitution_destination_length_csv = (
        args.substitution_destination_length_csv
        or error_input.with_name(
            f"{error_input.stem}_substitution_destination_length.csv"
        )
    )
    insertion_length_output = args.insertion_length_output or error_input.with_name(
        f"{error_input.stem}_insertion_destination_length.png"
    )
    insertion_length_csv = args.insertion_length_csv or error_input.with_name(
        f"{error_input.stem}_insertion_destination_length.csv"
    )

    deletion_length_counts = count_error_length_categories(
        error_input, "deletion", "source_word_length"
    )
    substitution_source_length_counts = count_error_length_categories(
        error_input, "substitution", "source_word_length"
    )
    substitution_destination_length_counts = count_error_length_categories(
        error_input, "substitution", "destination_word_length"
    )
    insertion_length_counts = count_error_length_categories(
        error_input, "insertion", "destination_word_length"
    )

    write_distribution_plot(
        deletion_length_counts,
        deletion_length_output,
        x_label="Source word length",
        title="Deletions by source word length",
    )
    write_counts_csv(
        deletion_length_counts, deletion_length_csv, "source_word_length"
    )
    write_distribution_plot(
        substitution_source_length_counts,
        substitution_source_length_output,
        x_label="Source word length",
        title="Substitutions by source word length",
    )
    write_counts_csv(
        substitution_source_length_counts,
        substitution_source_length_csv,
        "source_word_length",
    )
    write_distribution_plot(
        substitution_destination_length_counts,
        substitution_destination_length_output,
        x_label="Destination word length",
        title="Substitutions by destination word length",
    )
    write_counts_csv(
        substitution_destination_length_counts,
        substitution_destination_length_csv,
        "destination_word_length",
    )
    write_distribution_plot(
        insertion_length_counts,
        insertion_length_output,
        x_label="Destination word length",
        title="Insertions by destination word length",
    )
    write_counts_csv(
        insertion_length_counts,
        insertion_length_csv,
        "destination_word_length",
    )

    print(f"Wrote deletion source length plot to {deletion_length_output}")
    print(f"Wrote deletion source length table to {deletion_length_csv}")
    print(
        "Wrote substitution source length plot to "
        f"{substitution_source_length_output}"
    )
    print(
        "Wrote substitution source length table to "
        f"{substitution_source_length_csv}"
    )
    print(
        "Wrote substitution destination length plot to "
        f"{substitution_destination_length_output}"
    )
    print(
        "Wrote substitution destination length table to "
        f"{substitution_destination_length_csv}"
    )
    print(f"Wrote insertion destination length plot to {insertion_length_output}")
    print(f"Wrote insertion destination length table to {insertion_length_csv}")


if __name__ == "__main__":
    main()
