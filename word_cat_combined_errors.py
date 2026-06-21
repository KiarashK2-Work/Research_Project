from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "results" / "word_cat" / "hmi" / "whisper"
SPEECH_TYPE: str | None = None
MODEL_NAME: str | None = None

ERROR_SERIES = {
    "deletion_source": {
        "filename_suffix": "deletion_source_pos.csv",
        "tag_column": "source_pos",
        "label": "Deletion source",
    },
    "substitution_source": {
        "filename_suffix": "substitution_source_pos.csv",
        "tag_column": "source_pos",
        "label": "Substitution source",
    },
    "substitution_destination": {
        "filename_suffix": "substitution_destination_pos.csv",
        "tag_column": "destination_pos",
        "label": "Substitution destination",
    },
    "insertion_destination": {
        "filename_suffix": "insertion_destination_pos.csv",
        "tag_column": "destination_pos",
        "label": "Insertion destination",
    },
}


def infer_speech_type_label(input_dir: Path) -> str:
    speech_type = SPEECH_TYPE
    if speech_type is None:
        path_parts = [part.lower() for part in input_dir.parts]
        if "hmi" in path_parts:
            speech_type = "hmi"
        elif "read" in path_parts:
            speech_type = "read"
        else:
            speech_type = input_dir.parent.name

    if speech_type.lower() == "hmi":
        return "HMI"
    return speech_type


def infer_model_name(input_dir: Path) -> str:
    model_name = MODEL_NAME
    if model_name is None:
        path_parts = [part.lower() for part in input_dir.parts]
        if "google" in path_parts:
            model_name = "Chirp"
        elif "whisper" in path_parts:
            model_name = "Whisper"
        else:
            model_name = input_dir.name

    return model_name


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

    if not counts:
        raise ValueError(f"No count rows found in {input_file}")

    return counts


def load_error_counts(input_dir: Path) -> dict[str, dict[str, int]]:
    combined_counts: dict[str, dict[str, int]] = {}

    for series_name, config in ERROR_SERIES.items():
        input_file = find_error_csv(input_dir, str(config["filename_suffix"]))
        combined_counts[series_name] = read_counts(input_file, str(config["tag_column"]))

    return combined_counts


def write_combined_error_plot(
    combined_counts: dict[str, dict[str, int]],
    output_file: Path,
    title: str,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    tags = sorted(
        {
            tag
            for series_counts in combined_counts.values()
            for tag in series_counts
        }
    )
    if not tags:
        raise ValueError("No POS tags found to plot")

    bar_width = 0.2
    x_positions = list(range(len(tags)))
    figure_width = max(12, len(tags) * 0.65)

    plt.figure(figsize=(figure_width, 6))
    for index, (series_name, config) in enumerate(ERROR_SERIES.items()):
        offset = (index - (len(ERROR_SERIES) - 1) / 2) * bar_width
        values = [combined_counts[series_name].get(tag, 0) for tag in tags]
        positions = [position + offset for position in x_positions]
        plt.bar(
            positions,
            values,
            width=bar_width,
            label=str(config["label"]),
        )

    plt.xlabel("POS tag")
    plt.ylabel("Count")
    plt.title(title)
    plt.xticks(x_positions, tags, rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one combined plot for word-category error counts."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help=f"Directory containing word_cat_errors summary CSVs. Defaults to {INPUT_DIR}.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        help=(
            "Combined output PNG. Defaults to "
            "<input dir>/<word_cat_errors prefix>_combined_pos_errors.png."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Could not find input directory: {input_dir}")

    deletion_source_file = find_error_csv(input_dir, "deletion_source_pos.csv")
    output_plot = args.output_plot or deletion_source_file.with_name(
        deletion_source_file.name.replace(
            "_deletion_source_pos.csv",
            "_combined_pos_errors.png",
        )
    )

    combined_counts = load_error_counts(input_dir)
    write_combined_error_plot(
        combined_counts,
        output_plot,
        f"{infer_model_name(input_dir)} word-category errors for "
        f"{infer_speech_type_label(input_dir)} speech",
    )

    print(f"Wrote combined word-category error plot to {output_plot}")


if __name__ == "__main__":
    main()
