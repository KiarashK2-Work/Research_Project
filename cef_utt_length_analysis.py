from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = SCRIPT_DIR / "results" / "utt_length"
DEFAULT_SPEAKERS_FILE = SCRIPT_DIR / "data" / "speakers.txt"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "results" / "CEF" / "utt_length"

SPEECH_TYPES = ("read", "hmi")
INPUT_MODELS = ("google", "whisper")
MODEL_LABELS = {"google": "chirp", "whisper": "whisper"}
MODEL_TITLES = {"google": "Chirp", "whisper": "Whisper"}
CEF_LEVELS = ("A1", "A2", "B1")
METRICS = {
    "average_wer": "WER",
    "substitution_rate": "Substitution rate",
    "deletion_rate": "Deletion rate",
    "insertion_rate": "Insertion rate",
}
CONFIDENCE_Z = 1.96
CEF_COLORS = {"A1": "#1f77b4", "A2": "#ff7f0e", "B1": "#2ca02c"}
RATE_SCALE = 100.0
Y_LIMIT = (0.0, 100.0)
LABEL_FONT_SIZE = 24
TICK_FONT_SIZE = 20
LEGEND_FONT_SIZE = 20


def load_cef_by_speaker(speakers_file: Path) -> dict[str, str]:
    cef_by_speaker: dict[str, str] = {}
    with speakers_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required_columns = {"RegionSpeaker", "CEF"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{speakers_file} is missing required column(s): {missing}")

        for row in reader:
            speaker_id = (row["RegionSpeaker"] or "").strip().lower()
            cef = (row["CEF"] or "").strip().upper()
            if speaker_id and cef in CEF_LEVELS:
                cef_by_speaker[speaker_id] = cef

    return cef_by_speaker


def read_utterance_rows(input_file: Path, cef_by_speaker: dict[str, str]) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    with input_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {
            "utterance_id",
            "user_id",
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
            user_id = (row["user_id"] or "").strip().lower()
            cef = cef_by_speaker.get(user_id)
            if cef not in CEF_LEVELS:
                continue

            try:
                utterance_length = int(row["utterance_length"])
                substitutions = int(row["substitutions"])
                deletions = int(row["deletions"])
                insertions = int(row["insertions"])
                wer = float(row["wer"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid numeric value at {input_file}:{row_number}") from error

            if utterance_length <= 0:
                raise ValueError(
                    f"Invalid utterance_length at {input_file}:{row_number}: {utterance_length}"
                )

            rows.append(
                {
                    "utterance_id": row["utterance_id"],
                    "user_id": user_id,
                    "cef": cef,
                    "utterance_length": utterance_length,
                    "substitutions": substitutions,
                    "deletions": deletions,
                    "insertions": insertions,
                    "wer": round(wer, 6),
                    "substitution_rate": round(substitutions / utterance_length, 6),
                    "deletion_rate": round(deletions / utterance_length, 6),
                    "insertion_rate": round(insertions / utterance_length, 6),
                }
            )

    return rows


def mean_and_ci(values: list[float]) -> tuple[float, float, float]:
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, mean, mean

    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    standard_error = math.sqrt(variance) / math.sqrt(len(values))
    margin = CONFIDENCE_Z * standard_error
    return mean, max(0.0, mean - margin), mean + margin


def summarize_by_length(
    rows: Iterable[dict[str, str | int | float]], speech: str, model: str
) -> list[dict[str, str | int | float]]:
    metrics_by_group: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(
        lambda: {
            "wer": [],
            "substitution_rate": [],
            "deletion_rate": [],
            "insertion_rate": [],
        }
    )

    for row in rows:
        cef = str(row["cef"])
        utterance_length = int(row["utterance_length"])
        group = metrics_by_group[(cef, utterance_length)]
        group["wer"].append(float(row["wer"]))
        group["substitution_rate"].append(float(row["substitution_rate"]))
        group["deletion_rate"].append(float(row["deletion_rate"]))
        group["insertion_rate"].append(float(row["insertion_rate"]))

    summary_rows: list[dict[str, str | int | float]] = []
    for cef in CEF_LEVELS:
        cef_lengths = sorted(length for group_cef, length in metrics_by_group if group_cef == cef)
        for utterance_length in cef_lengths:
            group = metrics_by_group[(cef, utterance_length)]
            average_wer, average_wer_ci_lower, average_wer_ci_upper = mean_and_ci(group["wer"])
            substitution_rate, substitution_rate_ci_lower, substitution_rate_ci_upper = mean_and_ci(
                group["substitution_rate"]
            )
            deletion_rate, deletion_rate_ci_lower, deletion_rate_ci_upper = mean_and_ci(
                group["deletion_rate"]
            )
            insertion_rate, insertion_rate_ci_lower, insertion_rate_ci_upper = mean_and_ci(
                group["insertion_rate"]
            )
            summary_rows.append(
                {
                    "speech": speech,
                    "model": MODEL_LABELS[model],
                    "cef": cef,
                    "utterance_length": utterance_length,
                    "utterance_count": len(group["wer"]),
                    "average_wer": round(average_wer, 6),
                    "average_wer_ci_lower": round(average_wer_ci_lower, 6),
                    "average_wer_ci_upper": round(average_wer_ci_upper, 6),
                    "substitution_rate": round(substitution_rate, 6),
                    "substitution_rate_ci_lower": round(substitution_rate_ci_lower, 6),
                    "substitution_rate_ci_upper": round(substitution_rate_ci_upper, 6),
                    "deletion_rate": round(deletion_rate, 6),
                    "deletion_rate_ci_lower": round(deletion_rate_ci_lower, 6),
                    "deletion_rate_ci_upper": round(deletion_rate_ci_upper, 6),
                    "insertion_rate": round(insertion_rate, 6),
                    "insertion_rate_ci_lower": round(insertion_rate_ci_lower, 6),
                    "insertion_rate_ci_upper": round(insertion_rate_ci_upper, 6),
                }
            )

    return summary_rows


def summarize_length_distribution(
    rows: Iterable[dict[str, str | int | float]], speech: str, model: str
) -> list[dict[str, str | int | float]]:
    counts_by_cef: dict[str, Counter[int]] = {cef: Counter() for cef in CEF_LEVELS}
    totals_by_cef: Counter[str] = Counter()

    for row in rows:
        cef = str(row["cef"])
        utterance_length = int(row["utterance_length"])
        counts_by_cef[cef][utterance_length] += 1
        totals_by_cef[cef] += 1

    distribution_rows: list[dict[str, str | int | float]] = []
    for cef in CEF_LEVELS:
        total = totals_by_cef[cef]
        for utterance_length in sorted(counts_by_cef[cef]):
            count = counts_by_cef[cef][utterance_length]
            distribution_rows.append(
                {
                    "speech": speech,
                    "model": MODEL_LABELS[model],
                    "cef": cef,
                    "utterance_length": utterance_length,
                    "utterance_count": count,
                    "proportion": round(count / total, 6) if total else 0.0,
                    "percentage": round((count / total) * 100, 3) if total else 0.0,
                }
            )

    return distribution_rows


def write_csv(rows: list[dict[str, str | int | float]], output_file: Path) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {output_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_metric_csv(
    summary_rows: list[dict[str, str | int | float]], output_file: Path, metric: str
) -> None:
    rows = [
        {
            "speech": row["speech"],
            "model": row["model"],
            "cef": row["cef"],
            "utterance_length": row["utterance_length"],
            "utterance_count": row["utterance_count"],
            metric: row[metric],
            f"{metric}_ci_lower": row[f"{metric}_ci_lower"],
            f"{metric}_ci_upper": row[f"{metric}_ci_upper"],
        }
        for row in summary_rows
    ]
    write_csv(rows, output_file)


def plot_metric(
    summary_rows: list[dict[str, str | int | float]],
    output_file: Path,
    metric: str,
    speech: str,
    model: str,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))

    for cef in CEF_LEVELS:
        cef_rows = [row for row in summary_rows if row["cef"] == cef]
        if not cef_rows:
            continue

        lengths = [int(row["utterance_length"]) for row in cef_rows]
        values = [float(row[metric]) * RATE_SCALE for row in cef_rows]
        ci_lowers = [
            float(row[f"{metric}_ci_lower"]) * RATE_SCALE for row in cef_rows
        ]
        ci_uppers = [
            float(row[f"{metric}_ci_upper"]) * RATE_SCALE for row in cef_rows
        ]
        color = CEF_COLORS[cef]
        plt.plot(
            lengths,
            values,
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=cef,
            color=color,
        )
        plt.fill_between(
            lengths,
            ci_lowers,
            ci_uppers,
            color=color,
            alpha=0.13,
            linewidth=0,
        )

    y_label = "WER (%)" if metric == "average_wer" else f"{METRICS[metric]} (%)"
    plt.xlabel("Utterance length", fontsize=LABEL_FONT_SIZE)
    plt.ylabel(y_label, fontsize=LABEL_FONT_SIZE)
    plt.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    plt.ylim(*Y_LIMIT)
    plt.grid(True, alpha=0.3)
    legend = plt.legend(title="CEF", fontsize=LEGEND_FONT_SIZE)
    legend.get_title().set_fontsize(LEGEND_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()


def plot_length_distribution(
    distribution_rows: list[dict[str, str | int | float]],
    output_file: Path,
    speech: str,
    model: str,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))

    for cef in CEF_LEVELS:
        cef_rows = [row for row in distribution_rows if row["cef"] == cef]
        if not cef_rows:
            continue

        lengths = [int(row["utterance_length"]) for row in cef_rows]
        percentages = [float(row["percentage"]) for row in cef_rows]
        plt.plot(
            lengths,
            percentages,
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=cef,
            color=CEF_COLORS[cef],
        )

    plt.xlabel("Utterance length", fontsize=LABEL_FONT_SIZE)
    plt.ylabel("Utterances (%)", fontsize=LABEL_FONT_SIZE)
    plt.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    plt.ylim(*Y_LIMIT)
    plt.grid(True, alpha=0.3)
    legend = plt.legend(title="CEF", fontsize=LEGEND_FONT_SIZE)
    legend.get_title().set_fontsize(LEGEND_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare utterance-length ASR metrics across CEF levels."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Existing utterance-length result root. Defaults to {DEFAULT_INPUT_ROOT}.",
    )
    parser.add_argument(
        "--speakers-file",
        type=Path,
        default=DEFAULT_SPEAKERS_FILE,
        help=f"Speaker metadata with CEF labels. Defaults to {DEFAULT_SPEAKERS_FILE}.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output root. Defaults to {DEFAULT_OUTPUT_ROOT}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    speakers_file = args.speakers_file.resolve()
    output_root = args.output_root.resolve()

    cef_by_speaker = load_cef_by_speaker(speakers_file)
    all_summary_rows: list[dict[str, str | int | float]] = []
    all_distribution_rows: list[dict[str, str | int | float]] = []

    for speech in SPEECH_TYPES:
        for model in INPUT_MODELS:
            input_file = input_root / speech / model / f"utt_length_{speech}_{model}.csv"
            if not input_file.exists():
                raise FileNotFoundError(f"Could not find input file: {input_file}")

            output_dir = output_root / speech / MODEL_LABELS[model]
            output_prefix = f"cef_utt_length_{speech}_{MODEL_LABELS[model]}"

            utterance_rows = read_utterance_rows(input_file, cef_by_speaker)
            if not utterance_rows:
                raise ValueError(f"No A1/A2/B1 utterances found in {input_file}")

            summary_rows = summarize_by_length(utterance_rows, speech, model)
            distribution_rows = summarize_length_distribution(utterance_rows, speech, model)
            all_summary_rows.extend(summary_rows)
            all_distribution_rows.extend(distribution_rows)

            write_csv(utterance_rows, output_dir / f"{output_prefix}_utterances.csv")
            write_csv(summary_rows, output_dir / f"{output_prefix}_by_length.csv")
            write_csv(distribution_rows, output_dir / f"{output_prefix}_length_distribution.csv")

            for metric in METRICS:
                write_metric_csv(summary_rows, output_dir / f"{output_prefix}_{metric}_by_length.csv", metric)
                plot_metric(summary_rows, output_dir / f"{output_prefix}_{metric}_by_length.png", metric, speech, model)

            plot_length_distribution(
                distribution_rows,
                output_dir / f"{output_prefix}_length_distribution.png",
                speech,
                model,
            )

            counts = Counter(str(row["cef"]) for row in utterance_rows)
            print(
                f"{speech} {MODEL_LABELS[model]}: "
                + ", ".join(f"{cef}={counts[cef]}" for cef in CEF_LEVELS)
            )

    write_csv(all_summary_rows, output_root / "cef_utt_length_all_by_length.csv")
    write_csv(all_distribution_rows, output_root / "cef_utt_length_all_length_distribution.csv")
    print(f"Wrote CEF utterance-length outputs to {output_root}")


if __name__ == "__main__":
    main()
