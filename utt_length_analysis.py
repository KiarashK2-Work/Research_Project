from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "results" / "utt_length" / "hmi" / "google"
SPEECH_TYPE: str | None = None
MODEL_NAME: str | None = None
CONFIDENCE_Z = 1.96
RATE_SCALE = 100.0
RATE_Y_LIMIT = (0.0, 0.6)
COMBINED_RATE_Y_LIMIT = (0.0, 1.0)
DISTRIBUTION_LABEL_FONT_SIZE = 24
DISTRIBUTION_TICK_FONT_SIZE = 20


def find_default_input_file(input_dir: Path) -> Path:
    csv_files = sorted(
        path
        for path in input_dir.glob("*.csv")
        if not path.stem.endswith("_by_length")
    )
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")
    if len(csv_files) > 1:
        names = ", ".join(path.name for path in csv_files)
        raise ValueError(
            f"Found multiple CSV files in {input_dir}: {names}. "
            "Pass the file to analyze explicitly."
        )
    return csv_files[0]


def infer_speech_type_label(input_file: Path) -> str:
    speech_type = SPEECH_TYPE
    if speech_type is None:
        path_parts = [part.lower() for part in input_file.parts]
        if "hmi" in path_parts:
            speech_type = "hmi"
        elif "read" in path_parts:
            speech_type = "read"
        else:
            speech_type = input_file.parent.name

    if speech_type.lower() == "hmi":
        return "HMI"
    return speech_type


def infer_model_name(input_file: Path) -> str:
    model_name = MODEL_NAME
    if model_name is None:
        path_parts = [part.lower() for part in input_file.parts]
        if "google" in path_parts:
            model_name = "Chirp"
        elif "whisper" in path_parts:
            model_name = "Whisper"
        else:
            model_name = input_file.parent.name

    return model_name


def mean_and_ci(values: list[float]) -> tuple[float, float, float]:
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, mean, mean

    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    standard_error = math.sqrt(variance) / math.sqrt(len(values))
    margin = CONFIDENCE_Z * standard_error
    return mean, max(0.0, mean - margin), mean + margin


def group_by_utterance_length(input_file: Path) -> list[dict[str, int | float]]:
    metrics_by_length: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {
            "wer": [],
            "insertion_rate": [],
            "deletion_rate": [],
            "substitution_rate": [],
        }
    )

    with input_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {
            "utterance_length",
            "substitutions",
            "deletions",
            "insertions",
            "wer",
        }
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{input_file} is missing required column(s): {missing}")

        for row_number, row in enumerate(reader, start=2):
            try:
                utterance_length = int(row["utterance_length"])
                substitutions = int(row["substitutions"])
                deletions = int(row["deletions"])
                insertions = int(row["insertions"])
                wer = float(row["wer"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid numeric value at {input_file}:{row_number}"
                ) from error

            if utterance_length <= 0:
                raise ValueError(
                    f"Invalid utterance_length at {input_file}:{row_number}: "
                    f"{utterance_length}"
                )

            metrics_by_length[utterance_length]["wer"].append(wer)
            metrics_by_length[utterance_length]["insertion_rate"].append(
                insertions / utterance_length
            )
            metrics_by_length[utterance_length]["deletion_rate"].append(
                deletions / utterance_length
            )
            metrics_by_length[utterance_length]["substitution_rate"].append(
                substitutions / utterance_length
            )

    grouped_rows: list[dict[str, int | float]] = []
    for utterance_length in sorted(metrics_by_length):
        metrics = metrics_by_length[utterance_length]
        wers = metrics["wer"]
        average_wer, average_wer_ci_lower, average_wer_ci_upper = mean_and_ci(wers)
        insertion_rate, insertion_rate_ci_lower, insertion_rate_ci_upper = mean_and_ci(
            metrics["insertion_rate"]
        )
        deletion_rate, deletion_rate_ci_lower, deletion_rate_ci_upper = mean_and_ci(
            metrics["deletion_rate"]
        )
        (
            substitution_rate,
            substitution_rate_ci_lower,
            substitution_rate_ci_upper,
        ) = mean_and_ci(metrics["substitution_rate"])
        grouped_rows.append(
            {
                "utterance_length": utterance_length,
                "utterance_count": len(wers),
                "average_wer": round(average_wer, 6),
                "average_wer_ci_lower": round(average_wer_ci_lower, 6),
                "average_wer_ci_upper": round(average_wer_ci_upper, 6),
                "insertion_rate": round(insertion_rate, 6),
                "insertion_rate_ci_lower": round(insertion_rate_ci_lower, 6),
                "insertion_rate_ci_upper": round(insertion_rate_ci_upper, 6),
                "deletion_rate": round(deletion_rate, 6),
                "deletion_rate_ci_lower": round(deletion_rate_ci_lower, 6),
                "deletion_rate_ci_upper": round(deletion_rate_ci_upper, 6),
                "substitution_rate": round(substitution_rate, 6),
                "substitution_rate_ci_lower": round(substitution_rate_ci_lower, 6),
                "substitution_rate_ci_upper": round(substitution_rate_ci_upper, 6),
            }
        )

    return grouped_rows


def write_grouped_csv(rows: list[dict[str, int | float]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "utterance_length",
        "utterance_count",
        "average_wer",
        "average_wer_ci_lower",
        "average_wer_ci_upper",
        "insertion_rate",
        "insertion_rate_ci_lower",
        "insertion_rate_ci_upper",
        "deletion_rate",
        "deletion_rate_ci_lower",
        "deletion_rate_ci_upper",
        "substitution_rate",
        "substitution_rate_ci_lower",
        "substitution_rate_ci_upper",
    ]

    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_metric_csv(
    rows: list[dict[str, int | float]], output_file: Path, metric_name: str
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["utterance_length", metric_name]

    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "utterance_length": row["utterance_length"],
                    metric_name: row[metric_name],
                }
            )


def write_metric_plot(
    rows: list[dict[str, int | float]],
    output_file: Path,
    metric_name: str,
    y_label: str,
    title: str,
    y_limit: tuple[float, float] | None = None,
) -> None:
    if not rows:
        raise ValueError("No rows to plot")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    lengths = [int(row["utterance_length"]) for row in rows]
    values = [float(row[metric_name]) for row in rows]

    plt.figure(figsize=(10, 6))
    plt.plot(lengths, values, marker="o", linewidth=1.5, markersize=4)
    plt.xlabel("Utterance length")
    plt.ylabel(y_label)
    plt.title(title)
    if y_limit is not None:
        plt.ylim(*y_limit)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()


def write_combined_rate_plot(
    rows: list[dict[str, int | float]],
    output_file: Path,
    rate_labels: dict[str, str],
    title: str,
) -> None:
    if not rows:
        raise ValueError("No rows to plot")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    lengths = [int(row["utterance_length"]) for row in rows]

    plt.figure(figsize=(10, 6))
    wer_values = [float(row["average_wer"]) * RATE_SCALE for row in rows]
    wer_ci_lowers = [
        float(row["average_wer_ci_lower"]) * RATE_SCALE for row in rows
    ]
    wer_ci_uppers = [
        float(row["average_wer_ci_upper"]) * RATE_SCALE for row in rows
    ]
    wer_line = plt.plot(
        lengths,
        wer_values,
        marker="o",
        linewidth=1.8,
        markersize=4,
        label="WER",
    )[0]
    plt.fill_between(
        lengths,
        wer_ci_lowers,
        wer_ci_uppers,
        color=wer_line.get_color(),
        alpha=0.15,
        linewidth=0,
    )

    for metric_name, label in rate_labels.items():
        values = [float(row[metric_name]) * RATE_SCALE for row in rows]
        ci_lowers = [
            float(row[f"{metric_name}_ci_lower"]) * RATE_SCALE for row in rows
        ]
        ci_uppers = [
            float(row[f"{metric_name}_ci_upper"]) * RATE_SCALE for row in rows
        ]
        line = plt.plot(
            lengths,
            values,
            marker="o",
            linewidth=1.5,
            markersize=4,
            label=label,
        )[0]
        plt.fill_between(
            lengths,
            ci_lowers,
            ci_uppers,
            color=line.get_color(),
            alpha=0.15,
            linewidth=0,
        )

    plt.xlabel("Utterance length", fontsize=DISTRIBUTION_LABEL_FONT_SIZE)
    plt.ylabel("Rate (%)", fontsize=DISTRIBUTION_LABEL_FONT_SIZE)
    plt.tick_params(axis="both", labelsize=DISTRIBUTION_TICK_FONT_SIZE)
    plt.ylim(*(limit * RATE_SCALE for limit in COMBINED_RATE_Y_LIMIT))
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=DISTRIBUTION_TICK_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()


def write_length_distribution_plot(
    rows: list[dict[str, int | float]], output_file: Path, speech_type_label: str
) -> None:
    if not rows:
        raise ValueError("No rows to plot")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    lengths = [int(row["utterance_length"]) for row in rows]
    counts = [int(row["utterance_count"]) for row in rows]

    plt.figure(figsize=(10, 6))
    plt.bar(lengths, counts, width=0.8)
    plt.xlabel("Utterance length", fontsize=DISTRIBUTION_LABEL_FONT_SIZE)
    plt.ylabel("Count", fontsize=DISTRIBUTION_LABEL_FONT_SIZE)
    plt.tick_params(axis="both", labelsize=DISTRIBUTION_TICK_FONT_SIZE)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze average WER grouped by utterance length."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        type=Path,
        help="Input CSV. Defaults to the only CSV in results/utt_length/read.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Grouped output CSV. Defaults to <input stem>_by_length.csv.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        help="Output plot PNG. Defaults to <input stem>_by_length.png.",
    )
    parser.add_argument(
        "--output-distribution-plot",
        type=Path,
        help=(
            "Output utterance length distribution PNG. "
            "Defaults to <input stem>_length_distribution.png."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_file = args.input_file or find_default_input_file(INPUT_DIR)
    input_file = input_file.resolve()

    if not input_file.exists():
        raise FileNotFoundError(f"Could not find input file: {input_file}")

    speech_type_label = infer_speech_type_label(input_file)
    model_name = infer_model_name(input_file)

    output_csv = args.output_csv or input_file.with_name(f"{input_file.stem}_by_length.csv")
    output_plot = args.output_plot or input_file.with_name(
        f"{input_file.stem}_by_length.png"
    )
    output_distribution_plot = args.output_distribution_plot or input_file.with_name(
        f"{input_file.stem}_length_distribution.png"
    )
    output_combined_rate_plot = input_file.with_name(
        f"{input_file.stem}_error_rates_by_length.png"
    )

    rows = group_by_utterance_length(input_file)
    write_grouped_csv(rows, output_csv)
    write_metric_plot(
        rows,
        output_plot,
        "average_wer",
        "Average WER",
        f"{model_name} average WER by utterance length for "
        f"{speech_type_label} speech",
    )
    write_length_distribution_plot(rows, output_distribution_plot, speech_type_label)

    rate_outputs = {
        "insertion_rate": input_file.with_name(
            f"{input_file.stem}_insertion_rate_by_length.csv"
        ),
        "deletion_rate": input_file.with_name(
            f"{input_file.stem}_deletion_rate_by_length.csv"
        ),
        "substitution_rate": input_file.with_name(
            f"{input_file.stem}_substitution_rate_by_length.csv"
        ),
    }
    rate_plot_outputs = {
        "insertion_rate": input_file.with_name(
            f"{input_file.stem}_insertion_rate_by_length.png"
        ),
        "deletion_rate": input_file.with_name(
            f"{input_file.stem}_deletion_rate_by_length.png"
        ),
        "substitution_rate": input_file.with_name(
            f"{input_file.stem}_substitution_rate_by_length.png"
        ),
    }
    rate_labels = {
        "insertion_rate": "Insertion rate",
        "deletion_rate": "Deletion rate",
        "substitution_rate": "Substitution rate",
    }

    for metric_name, rate_output in rate_outputs.items():
        write_metric_csv(rows, rate_output, metric_name)
        write_metric_plot(
            rows,
            rate_plot_outputs[metric_name],
            metric_name,
            rate_labels[metric_name],
            f"{model_name} {rate_labels[metric_name].lower()} by utterance length for "
            f"{speech_type_label} speech",
            RATE_Y_LIMIT,
        )

    write_combined_rate_plot(
        rows,
        output_combined_rate_plot,
        rate_labels,
        f"{model_name} WER, insertion, deletion, and substitution rates by "
        f"utterance length for {speech_type_label} speech",
    )

    print(f"Wrote grouped table to {output_csv}")
    print(f"Wrote WER plot to {output_plot}")
    print(f"Wrote length distribution plot to {output_distribution_plot}")
    for metric_name, rate_output in rate_outputs.items():
        print(f"Wrote {metric_name} table to {rate_output}")
        print(f"Wrote {metric_name} plot to {rate_plot_outputs[metric_name]}")
    print(f"Wrote combined error rates plot to {output_combined_rate_plot}")


if __name__ == "__main__":
    main()
