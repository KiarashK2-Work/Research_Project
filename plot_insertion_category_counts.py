from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results" / "word_cat"
OUTPUT_DIR = RESULTS_DIR / "insertion_category_counts"
EXCLUDED_TAGS = {"X"}
Y_AXIS_MAX = 350
PERCENTAGE_Y_AXIS_MAX = 25
LABEL_FONT_SIZE = 24
X_TICK_FONT_SIZE = 14
Y_TICK_FONT_SIZE = 20
LEGEND_FONT_SIZE = 20

ASR_SYSTEMS = {
    "chirp": {
        "result_dir": "google",
        "label": "Chirp",
    },
    "whisper": {
        "result_dir": "whisper",
        "label": "Whisper",
    },
}
SPEECH_TYPES = {
    "read": {
        "result_dir": "read",
        "label": "read",
    },
    "hmi": {
        "result_dir": "hmi",
        "label": "HMI",
    },
}


def find_insertion_csv(input_dir: Path) -> Path:
    matches = sorted(input_dir.glob("*_insertion_destination_pos.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No *_insertion_destination_pos.csv file found in {input_dir}"
        )
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise ValueError(
            f"Found multiple insertion destination files in {input_dir}: {names}"
        )
    return matches[0]


def read_counts(input_file: Path) -> tuple[dict[str, int], int]:
    counts: dict[str, int] = {}
    total_insertions = 0

    with input_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"destination_pos", "count"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{input_file} is missing required column(s): {missing}")

        for row_number, row in enumerate(reader, start=2):
            destination_pos = row["destination_pos"]
            if destination_pos == "TOTAL":
                continue

            try:
                count = int(row["count"])
            except ValueError as error:
                raise ValueError(
                    f"Invalid count at {input_file}:{row_number}"
                ) from error

            total_insertions += count
            counts[destination_pos] = count

    return counts, total_insertions


def load_counts(
    asr_config: dict[str, str],
) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    counts_by_speech: dict[str, dict[str, int]] = {}
    totals_by_speech: dict[str, int] = {}

    for speech_key, speech_config in SPEECH_TYPES.items():
        input_dir = (
            RESULTS_DIR / speech_config["result_dir"] / asr_config["result_dir"]
        )
        input_file = find_insertion_csv(input_dir)
        counts_by_speech[speech_key], totals_by_speech[speech_key] = read_counts(
            input_file
        )

    return counts_by_speech, totals_by_speech


def percentage_counts(
    counts_by_speech: dict[str, dict[str, int]],
    totals_by_speech: dict[str, int],
) -> dict[str, dict[str, float]]:
    percentages_by_speech: dict[str, dict[str, float]] = {}

    for speech_key, speech_counts in counts_by_speech.items():
        total_insertions = totals_by_speech[speech_key]
        if total_insertions == 0:
            percentages_by_speech[speech_key] = {
                tag: 0.0 for tag in speech_counts
            }
            continue

        percentages_by_speech[speech_key] = {
            tag: count / total_insertions * 100
            for tag, count in speech_counts.items()
        }

    return percentages_by_speech


def filter_plot_tags(
    values_by_speech: dict[str, dict[str, int | float]],
) -> dict[str, dict[str, int | float]]:
    return {
        speech_key: {
            tag: value
            for tag, value in speech_values.items()
            if tag not in EXCLUDED_TAGS
        }
        for speech_key, speech_values in values_by_speech.items()
    }


def write_insertion_table(
    counts_by_speech: dict[str, dict[str, int]],
    percentages_by_speech: dict[str, dict[str, float]],
    totals_by_speech: dict[str, int],
    model: str,
    output_file: Path,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tags = sorted(
        {
            tag
            for speech_counts in counts_by_speech.values()
            for tag in speech_counts
        }
        | EXCLUDED_TAGS
    )

    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "speech_type",
                "destination_pos",
                "count",
                "percentage",
                "total_insertions",
            ],
        )
        writer.writeheader()

        for speech_key, speech_config in SPEECH_TYPES.items():
            for tag in tags:
                count = counts_by_speech[speech_key].get(tag, 0)
                percentage = percentages_by_speech[speech_key].get(tag, 0.0)
                writer.writerow(
                    {
                        "model": model,
                        "speech_type": speech_config["label"],
                        "destination_pos": tag,
                        "count": count,
                        "percentage": f"{percentage:.4f}",
                        "total_insertions": totals_by_speech[speech_key],
                    }
                )


def write_grouped_count_plot(
    counts_by_speech: dict[str, dict[str, int | float]],
    output_file: Path,
    title: str,
    y_label: str,
    y_axis_max: float,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tags = sorted(
        {
            tag
            for speech_counts in counts_by_speech.values()
            for tag in speech_counts
        }
    )
    if not tags:
        raise ValueError(f"No insertion counts found for {title}")

    bar_width = 0.38
    x_positions = list(range(len(tags)))
    figure_width = max(12, len(tags) * 0.7)

    plt.figure(figsize=(figure_width, 6))
    for index, (speech_key, speech_config) in enumerate(SPEECH_TYPES.items()):
        offset = (index - 0.5) * bar_width
        values = [counts_by_speech[speech_key].get(tag, 0) for tag in tags]
        positions = [position + offset for position in x_positions]
        plt.bar(
            positions,
            values,
            width=bar_width,
            label=speech_config["label"],
        )

    plt.xlabel("Destination POS tag", fontsize=LABEL_FONT_SIZE)
    plt.ylabel(y_label, fontsize=LABEL_FONT_SIZE)
    plt.ylim(0, y_axis_max)
    plt.xticks(x_positions, tags, rotation=45, ha="right", fontsize=X_TICK_FONT_SIZE)
    plt.tick_params(axis="y", labelsize=Y_TICK_FONT_SIZE)
    plt.grid(axis="y", alpha=0.3)
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()


def main() -> None:
    for asr_key, asr_config in ASR_SYSTEMS.items():
        counts_by_speech, totals_by_speech = load_counts(asr_config)
        percentages_by_speech = percentage_counts(
            counts_by_speech, totals_by_speech
        )

        output_file = OUTPUT_DIR / f"{asr_key}_insertion_destination_table.csv"
        write_insertion_table(
            counts_by_speech,
            percentages_by_speech,
            totals_by_speech,
            asr_key,
            output_file,
        )
        print(f"Wrote insertion table to {output_file}")

        output_file = OUTPUT_DIR / f"{asr_key}_insertion_destination_counts.png"
        write_grouped_count_plot(
            filter_plot_tags(counts_by_speech),
            output_file,
            f"{asr_config['label']} insertion destination counts",
            "Insertion count",
            Y_AXIS_MAX,
        )
        print(f"Wrote insertion count plot to {output_file}")

        output_file = (
            OUTPUT_DIR / f"{asr_key}_insertion_destination_percentages.png"
        )
        write_grouped_count_plot(
            filter_plot_tags(percentages_by_speech),
            output_file,
            f"{asr_config['label']} insertion destination percentages",
            "Percentage of total insertions",
            PERCENTAGE_Y_AXIS_MAX,
        )
        print(f"Wrote insertion percentage plot to {output_file}")


if __name__ == "__main__":
    main()
