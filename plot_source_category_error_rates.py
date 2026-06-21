from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results" / "word_cat"
OUTPUT_DIR = RESULTS_DIR / "source_category_error_rates"
RATE_SCALE = 100.0
Y_AXIS_MAX = 1.0
CONFIDENCE_Z = 1.96
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
ERROR_TYPES = {
    "deletion": {
        "filename_suffix": "deletion_source_pos.csv",
        "title_label": "deletion source",
    },
    "substitution": {
        "filename_suffix": "substitution_source_pos.csv",
        "title_label": "substitution source",
    },
}
EXCLUDED_TAGS = {"PUNCT"}
DELETION_EXTRA_EXCLUDED_TAGS = {"SYM", "X"}


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


def find_ref_pos_counts_csv(input_dir: Path) -> Path:
    matches = sorted(input_dir.glob("ref_pos_counts_*.csv"))
    if not matches:
        raise FileNotFoundError(f"No ref_pos_counts_*.csv file found in {input_dir}")
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise ValueError(
            f"Found multiple ref_pos_counts_*.csv files in {input_dir}: {names}"
        )
    return matches[0]


def read_ref_pos_counts(input_file: Path) -> dict[str, int]:
    counts: dict[str, int] = {}

    with input_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"pos_tag", "count"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{input_file} is missing required column(s): {missing}")

        for row_number, row in enumerate(reader, start=2):
            pos_tag = row["pos_tag"]
            if pos_tag == "TOTAL":
                continue

            try:
                counts[pos_tag] = int(row["count"])
            except ValueError as error:
                raise ValueError(
                    f"Invalid count at {input_file}:{row_number}"
                ) from error

    return counts


def proportion_ci(successes: int, total: int) -> tuple[float, float, float]:
    if total == 0:
        return 0.0, 0.0, 0.0

    proportion = successes / total
    standard_error = (proportion * (1.0 - proportion) / total) ** 0.5
    margin = CONFIDENCE_Z * standard_error
    return proportion, max(0.0, proportion - margin), min(1.0, proportion + margin)


def read_error_rates(
    input_file: Path,
    reference_counts: dict[str, int],
    excluded_tags: set[str],
) -> dict[str, tuple[float, float, float]]:
    rates: dict[str, tuple[float, float, float]] = {}

    with input_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"source_pos", "count"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{input_file} is missing required column(s): {missing}")

        for row_number, row in enumerate(reader, start=2):
            source_pos = row["source_pos"]
            if source_pos == "TOTAL" or source_pos in excluded_tags:
                continue
            reference_count = reference_counts.get(source_pos, 0)
            if reference_count == 0:
                continue

            try:
                error_count = int(row["count"])
            except ValueError as error:
                raise ValueError(
                    f"Invalid count at {input_file}:{row_number}"
                ) from error
            rates[source_pos] = proportion_ci(error_count, reference_count)

    return rates


def load_rates(
    asr_config: dict[str, str],
    error_config: dict[str, str],
    excluded_tags: set[str],
):
    rates_by_speech: dict[str, dict[str, tuple[float, float, float]]] = {}

    for speech_key, speech_config in SPEECH_TYPES.items():
        input_dir = (
            RESULTS_DIR / speech_config["result_dir"] / asr_config["result_dir"]
        )
        input_file = find_error_csv(input_dir, error_config["filename_suffix"])
        reference_counts = read_ref_pos_counts(find_ref_pos_counts_csv(input_dir))
        rates_by_speech[speech_key] = read_error_rates(
            input_file, reference_counts, excluded_tags
        )

    return rates_by_speech


def write_grouped_rate_plot(
    rates_by_speech: dict[str, dict[str, tuple[float, float, float]]],
    output_file: Path,
    title: str,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tags = sorted(
        {
            tag
            for speech_rates in rates_by_speech.values()
            for tag in speech_rates
        }
    )
    if not tags:
        raise ValueError(f"No category error rates found for {title}")

    bar_width = 0.38
    x_positions = list(range(len(tags)))
    figure_width = max(12, len(tags) * 0.7)

    plt.figure(figsize=(figure_width, 6))
    for index, (speech_key, speech_config) in enumerate(SPEECH_TYPES.items()):
        offset = (index - 0.5) * bar_width
        intervals = [
            rates_by_speech[speech_key].get(tag, (0.0, 0.0, 0.0)) for tag in tags
        ]
        values = [interval[0] * RATE_SCALE for interval in intervals]
        lower_errors = [
            value - interval[1] * RATE_SCALE
            for value, interval in zip(values, intervals)
        ]
        upper_errors = [
            interval[2] * RATE_SCALE - value
            for value, interval in zip(values, intervals)
        ]
        positions = [position + offset for position in x_positions]
        plt.errorbar(
            positions,
            values,
            yerr=[lower_errors, upper_errors],
            fmt="o",
            capsize=4,
            linestyle="none",
            label=speech_config["label"],
        )

    plt.xlabel("Source POS tag", fontsize=LABEL_FONT_SIZE)
    plt.ylabel("Category error rate (%)", fontsize=LABEL_FONT_SIZE)
    plt.ylim(0, Y_AXIS_MAX * RATE_SCALE)
    plt.xticks(x_positions, tags, rotation=45, ha="right", fontsize=X_TICK_FONT_SIZE)
    plt.tick_params(axis="y", labelsize=Y_TICK_FONT_SIZE)
    plt.grid(axis="y", alpha=0.3)
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()


def main() -> None:
    for asr_key, asr_config in ASR_SYSTEMS.items():
        for error_key, error_config in ERROR_TYPES.items():
            rates_by_speech = load_rates(asr_config, error_config, EXCLUDED_TAGS)
            output_file = (
                OUTPUT_DIR / f"{asr_key}_{error_key}_source_category_error_rates.png"
            )
            write_grouped_rate_plot(
                rates_by_speech,
                output_file,
                f"{asr_config['label']} {error_config['title_label']} rates",
            )
            print(f"Wrote category error rate plot to {output_file}")

            if error_key == "deletion":
                excluded_tags = EXCLUDED_TAGS | DELETION_EXTRA_EXCLUDED_TAGS
                rates_by_speech = load_rates(asr_config, error_config, excluded_tags)
                output_file = (
                    OUTPUT_DIR
                    / f"{asr_key}_{error_key}_source_category_error_rates_no_sym_x.png"
                )
                write_grouped_rate_plot(
                    rates_by_speech,
                    output_file,
                    f"{asr_config['label']} {error_config['title_label']} rates",
                )
                print(f"Wrote category error rate plot to {output_file}")


if __name__ == "__main__":
    main()
