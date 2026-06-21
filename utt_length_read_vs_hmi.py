from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL = "whisper"
READ_INPUT = (
    SCRIPT_DIR
    / "results"
    / "utt_length"
    / "read"
    / MODEL
    / f"utt_length_read_{MODEL}.csv"
)
HMI_INPUT = (
    SCRIPT_DIR
    / "results"
    / "utt_length"
    / "hmi"
    / MODEL
    / f"utt_length_hmi_{MODEL}.csv"
)
OUTPUT_DIR = SCRIPT_DIR / "results" / "utt_length" / "read_vs_hmi" / MODEL
OUTPUT_CSV = OUTPUT_DIR / f"utt_length_read_vs_hmi_{MODEL}.csv"
OUTPUT_PLOT = OUTPUT_DIR / f"utt_length_read_vs_hmi_{MODEL}.png"
CONFIDENCE_Z = 1.96
WER_Y_LIMIT = (0.0, 1.0)


def model_title(model: str) -> str:
    if model.lower() == "google":
        return "Chirp"
    if model.lower() == "whisper":
        return "Whisper"
    return model


def read_wer_by_length(input_file: Path) -> dict[int, list[float]]:
    wer_by_length: dict[int, list[float]] = defaultdict(list)

    with input_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"utterance_length", "wer"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{input_file} is missing required column(s): {missing}")

        for row_number, row in enumerate(reader, start=2):
            try:
                utterance_length = int(row["utterance_length"])
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

            wer_by_length[utterance_length].append(wer)

    if not wer_by_length:
        raise ValueError(f"No WER rows found in {input_file}")

    return dict(wer_by_length)


def summarize_wer(
    wer_by_length: dict[int, list[float]], speech: str
) -> list[dict[str, int | str | float]]:
    rows: list[dict[str, int | str | float]] = []

    for utterance_length in sorted(wer_by_length):
        wers = wer_by_length[utterance_length]
        count = len(wers)
        mean_wer = sum(wers) / count
        if count > 1:
            variance = sum((wer - mean_wer) ** 2 for wer in wers) / (count - 1)
            standard_error = math.sqrt(variance) / math.sqrt(count)
        else:
            standard_error = 0.0

        margin = CONFIDENCE_Z * standard_error
        rows.append(
            {
                "speech": speech,
                "utterance_length": utterance_length,
                "utterance_count": count,
                "mean_wer": round(mean_wer, 6),
                "ci_lower": round(max(0.0, mean_wer - margin), 6),
                "ci_upper": round(mean_wer + margin, 6),
            }
        )

    return rows


def write_summary_csv(rows: list[dict[str, int | str | float]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "speech",
        "utterance_length",
        "utterance_count",
        "mean_wer",
        "ci_lower",
        "ci_upper",
    ]

    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_read_vs_hmi_plot(
    rows: list[dict[str, int | str | float]], output_file: Path, model: str
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    rows_by_speech = {
        speech: [row for row in rows if row["speech"] == speech]
        for speech in ("read", "HMI")
    }

    plt.figure(figsize=(10, 6))
    for speech, speech_rows in rows_by_speech.items():
        lengths = [int(row["utterance_length"]) for row in speech_rows]
        mean_wers = [float(row["mean_wer"]) for row in speech_rows]
        ci_lowers = [float(row["ci_lower"]) for row in speech_rows]
        ci_uppers = [float(row["ci_upper"]) for row in speech_rows]

        line = plt.plot(
            lengths,
            mean_wers,
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=speech,
        )[0]
        plt.fill_between(
            lengths,
            ci_lowers,
            ci_uppers,
            color=line.get_color(),
            alpha=0.18,
            linewidth=0,
        )

    plt.xlabel("Utterance length")
    plt.ylabel("Mean WER")
    plt.title(f"{model_title(model)} WER by utterance length for read and HMI speech")
    plt.ylim(*WER_Y_LIMIT)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare read and HMI WER by utterance length with confidence intervals."
    )
    parser.add_argument(
        "--read-input",
        type=Path,
        default=READ_INPUT,
        help=f"Read utterance-length CSV. Defaults to {READ_INPUT}.",
    )
    parser.add_argument(
        "--hmi-input",
        type=Path,
        default=HMI_INPUT,
        help=f"HMI utterance-length CSV. Defaults to {HMI_INPUT}.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=OUTPUT_CSV,
        help=f"Summary CSV for plotting. Defaults to {OUTPUT_CSV}.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=OUTPUT_PLOT,
        help=f"Output plot PNG. Defaults to {OUTPUT_PLOT}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    read_input = args.read_input.resolve()
    hmi_input = args.hmi_input.resolve()

    if not read_input.exists():
        raise FileNotFoundError(f"Could not find read input file: {read_input}")
    if not hmi_input.exists():
        raise FileNotFoundError(f"Could not find HMI input file: {hmi_input}")

    rows = [
        *summarize_wer(read_wer_by_length(read_input), "read"),
        *summarize_wer(read_wer_by_length(hmi_input), "HMI"),
    ]

    write_summary_csv(rows, args.output_csv.resolve())
    write_read_vs_hmi_plot(rows, args.output_plot.resolve(), MODEL)

    print(f"Wrote read vs HMI summary CSV to {args.output_csv.resolve()}")
    print(f"Wrote read vs HMI WER plot to {args.output_plot.resolve()}")


if __name__ == "__main__":
    main()
