from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results" / "word_cat"
OUTPUT_DIR = RESULTS_DIR / "source_category_error_rates"
GROUP_ORDER = ["open_class", "closed_class", "other"]
RATE_SCALE = 100.0
Y_AXIS_MAX = 0.7
CONFIDENCE_Z = 1.96
LABEL_FONT_SIZE = 20
TICK_FONT_SIZE = 20
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


def find_ref_group_counts_csv(
    input_dir: Path, speech_key: str, asr_key: str
) -> Path:
    counts_file = input_dir / f"ref_pos_group_counts_{speech_key}_{asr_key}.csv"
    if not counts_file.exists():
        raise FileNotFoundError(
            f"Could not find aggregated reference counts CSV: {counts_file}. "
            "Run aggregate_pos_group_distributions.py first."
        )
    return counts_file


def format_rate(error_count: int, reference_count: int) -> str:
    if reference_count == 0:
        return ""
    return f"{error_count / reference_count:.6f}"


def proportion_ci(error_count: int, reference_count: int) -> tuple[str, str]:
    if reference_count == 0:
        return "", ""

    proportion = error_count / reference_count
    standard_error = (proportion * (1.0 - proportion) / reference_count) ** 0.5
    margin = CONFIDENCE_Z * standard_error
    return f"{max(0.0, proportion - margin):.6f}", f"{min(1.0, proportion + margin):.6f}"


def read_group_error_counts(input_file: Path) -> dict[str, int]:
    counts = {group: 0 for group in GROUP_ORDER}

    with input_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"source_pos", "count"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{input_file} is missing required column(s): {missing}")

        for row_number, row in enumerate(reader, start=2):
            source_pos = row["source_pos"]
            if source_pos == "TOTAL":
                continue

            try:
                pos_group = POS_GROUPS[source_pos]
            except KeyError as error:
                raise ValueError(
                    f"No aggregate group configured for POS tag {source_pos!r} "
                    f"in {input_file}"
                ) from error

            try:
                counts[pos_group] += int(row["count"])
            except ValueError as error:
                raise ValueError(
                    f"Invalid count at {input_file}:{row_number}"
                ) from error

    return counts


def read_group_reference_counts(input_file: Path) -> dict[str, int]:
    counts: dict[str, int] = {}

    with input_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"pos_group", "count"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{input_file} is missing required column(s): {missing}")

        for row_number, row in enumerate(reader, start=2):
            pos_group = row["pos_group"]
            if pos_group == "TOTAL":
                continue
            try:
                counts[pos_group] = int(row["count"])
            except ValueError as error:
                raise ValueError(
                    f"Invalid count at {input_file}:{row_number}"
                ) from error

    return counts


def build_rows(
    asr_key: str, asr_config: dict[str, str], error_config: dict[str, str]
) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []

    for speech_key, speech_config in SPEECH_TYPES.items():
        input_dir = RESULTS_DIR / speech_config["result_dir"] / asr_config["result_dir"]
        error_file = find_error_csv(input_dir, error_config["filename_suffix"])
        ref_counts_file = find_ref_group_counts_csv(input_dir, speech_key, asr_config["result_dir"])
        error_counts = read_group_error_counts(error_file)
        reference_counts = read_group_reference_counts(ref_counts_file)

        for pos_group in GROUP_ORDER:
            error_count = error_counts.get(pos_group, 0)
            reference_count = reference_counts.get(pos_group, 0)
            ci_lower, ci_upper = proportion_ci(error_count, reference_count)
            rows.append(
                {
                    "model": asr_key,
                    "speech_type": speech_config["label"],
                    "pos_group": pos_group,
                    "error_count": error_count,
                    "reference_count": reference_count,
                    "category_error_rate": format_rate(
                        error_count, reference_count
                    ),
                    "category_error_rate_ci_lower": ci_lower,
                    "category_error_rate_ci_upper": ci_upper,
                }
            )

    return rows


def write_table(rows: list[dict[str, str | int]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "speech_type",
        "pos_group",
        "error_count",
        "reference_count",
        "category_error_rate",
        "category_error_rate_ci_lower",
        "category_error_rate_ci_upper",
    ]

    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_rate_plot(
    rows: list[dict[str, str | int]],
    output_file: Path,
    title: str,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    rates_by_speech = {
        row["speech_type"]: {
            existing_row["pos_group"]: (
                float(existing_row["category_error_rate"]),
                float(existing_row["category_error_rate_ci_lower"]),
                float(existing_row["category_error_rate_ci_upper"]),
            )
            for existing_row in rows
            if existing_row["speech_type"] == row["speech_type"]
            and existing_row["category_error_rate"] != ""
        }
        for row in rows
    }

    bar_width = 0.38
    x_positions = list(range(len(GROUP_ORDER)))

    plt.figure(figsize=(8, 6))
    for index, (speech_key, speech_config) in enumerate(SPEECH_TYPES.items()):
        offset = (index - 0.5) * bar_width
        speech_label = speech_config["label"]
        intervals = [
            rates_by_speech.get(speech_label, {}).get(pos_group, (0.0, 0.0, 0.0))
            for pos_group in GROUP_ORDER
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
            markersize=8,
            elinewidth=2,
            capsize=6,
            capthick=2,
            linestyle="none",
            label=speech_label,
        )

    plt.xlabel("Source aggregated word category", fontsize=LABEL_FONT_SIZE)
    plt.ylabel("Category error rate (%)", fontsize=LABEL_FONT_SIZE)
    plt.ylim(0, Y_AXIS_MAX * RATE_SCALE)
    plt.xticks(x_positions, GROUP_ORDER, rotation=20, ha="right", fontsize=TICK_FONT_SIZE)
    plt.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
    plt.grid(axis="y", alpha=0.3)
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()


def main() -> None:
    for asr_key, asr_config in ASR_SYSTEMS.items():
        for error_key, error_config in ERROR_TYPES.items():
            rows = build_rows(asr_key, asr_config, error_config)
            output_stem = f"{asr_key}_{error_key}_source_group_error_rates"
            table_file = OUTPUT_DIR / f"{output_stem}.csv"
            plot_file = OUTPUT_DIR / f"{output_stem}.png"

            write_table(rows, table_file)
            write_rate_plot(
                rows,
                plot_file,
                f"{asr_config['label']} {error_config['title_label']} group rates",
            )

            print(f"Wrote source group error-rate table to {table_file}")
            print(f"Wrote source group error-rate plot to {plot_file}")


if __name__ == "__main__":
    main()
