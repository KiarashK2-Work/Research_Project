from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results" / "CEF" / "word_cat"
CEF_LEVELS = ("A1", "A2", "B1")
CEF_COLORS = {"A1": "#1f77b4", "A2": "#ff7f0e", "B1": "#2ca02c"}
POS_GROUP_ORDER = ("open_class", "closed_class", "other")
RATE_SCALE = 100.0
Y_AXIS_MAX = 1.0
CONFIDENCE_Z = 1.96
LABEL_FONT_SIZE = 24
X_TICK_FONT_SIZE = 14
Y_TICK_FONT_SIZE = 20
LEGEND_FONT_SIZE = 20
EXCLUDED_TAGS = {"PUNCT"}

SPEECH_TYPES = {
    "read": {"label": "read"},
    "hmi": {"label": "HMI"},
}
ASR_SYSTEMS = {
    "chirp": {"label": "Chirp"},
    "whisper": {"label": "Whisper"},
}
ERROR_TYPES = {
    "deletion": {"title_label": "deletion source"},
    "substitution": {"title_label": "substitution source"},
}


def input_dir_for(speech_key: str, asr_key: str) -> Path:
    return RESULTS_DIR / speech_key / asr_key


def output_dir_for(speech_key: str) -> Path:
    return RESULTS_DIR / speech_key / "source_category_error_rates"


def input_file_for(speech_key: str, asr_key: str, stem: str) -> Path:
    return input_dir_for(speech_key, asr_key) / f"cef_word_cat_{speech_key}_{asr_key}_{stem}.csv"


def proportion_ci(successes: int, total: int) -> tuple[float, float, float]:
    if total == 0:
        return 0.0, 0.0, 0.0

    proportion = successes / total
    standard_error = (proportion * (1.0 - proportion) / total) ** 0.5
    margin = CONFIDENCE_Z * standard_error
    return proportion, max(0.0, proportion - margin), min(1.0, proportion + margin)


def read_ref_pos_counts(input_file: Path) -> dict[str, dict[str, int]]:
    counts = {cef: {} for cef in CEF_LEVELS}

    with input_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"cef", "pos_tag", "count"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{input_file} is missing required column(s): {missing}")

        for row_number, row in enumerate(reader, start=2):
            cef = row["cef"]
            pos_tag = row["pos_tag"]
            if cef not in CEF_LEVELS or pos_tag in EXCLUDED_TAGS:
                continue

            try:
                counts[cef][pos_tag] = int(row["count"])
            except ValueError as error:
                raise ValueError(
                    f"Invalid count at {input_file}:{row_number}"
                ) from error

    return counts


def read_error_counts(input_file: Path) -> dict[str, dict[str, int]]:
    counts = {cef: {} for cef in CEF_LEVELS}

    with input_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"cef", "source_pos", "count"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{input_file} is missing required column(s): {missing}")

        for row_number, row in enumerate(reader, start=2):
            cef = row["cef"]
            source_pos = row["source_pos"]
            if cef not in CEF_LEVELS or source_pos in EXCLUDED_TAGS:
                continue

            try:
                counts[cef][source_pos] = int(row["count"])
            except ValueError as error:
                raise ValueError(
                    f"Invalid count at {input_file}:{row_number}"
                ) from error

    return counts


def build_rate_rows(
    speech_key: str,
    asr_key: str,
    error_key: str,
) -> list[dict[str, str | int | float]]:
    ref_counts = read_ref_pos_counts(input_file_for(speech_key, asr_key, "ref_pos_counts"))
    error_counts = read_error_counts(
        input_file_for(speech_key, asr_key, f"{error_key}_source_pos_counts")
    )
    tags = sorted(
        {
            tag
            for cef_counts in ref_counts.values()
            for tag, reference_count in cef_counts.items()
            if reference_count > 0
        }
    )

    rows: list[dict[str, str | int | float]] = []
    for cef in CEF_LEVELS:
        for tag in tags:
            errors = error_counts[cef].get(tag, 0)
            references = ref_counts[cef].get(tag, 0)
            rate, lower, upper = proportion_ci(errors, references)
            rows.append(
                {
                    "speech_type": SPEECH_TYPES[speech_key]["label"],
                    "model": asr_key,
                    "cef": cef,
                    "error_type": error_key,
                    "source_pos": tag,
                    "error_count": errors,
                    "reference_count": references,
                    "category_error_rate": round(rate, 6),
                    "category_error_rate_ci_lower": round(lower, 6),
                    "category_error_rate_ci_upper": round(upper, 6),
                }
            )

    return rows


def read_source_group_rate_rows(input_file: Path) -> list[dict[str, str]]:
    with input_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {
            "cef",
            "pos_group",
            "category_error_rate",
            "category_error_rate_ci_lower",
            "category_error_rate_ci_upper",
        }
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{input_file} is missing required column(s): {missing}")

        return list(reader)


def write_csv(rows: list[dict[str, str | int | float]], output_file: Path) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {output_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_grouped_rate_plot(
    rows: list[dict[str, str | int | float]],
    output_file: Path,
    title: str,
    category_column: str,
    *,
    x_label: str,
    order: tuple[str, ...] | None = None,
    figure_size: tuple[float, float] | None = None,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    categories = list(order) if order is not None else sorted(
        {str(row[category_column]) for row in rows}
    )
    if not categories:
        raise ValueError(f"No category error rates found for {title}")

    bar_width = 0.24
    x_positions = list(range(len(categories)))
    figure_width = max(12, len(categories) * 0.7)

    plt.figure(figsize=figure_size or (figure_width, 6))
    for index, cef in enumerate(CEF_LEVELS):
        offset = (index - 1) * bar_width
        row_by_category = {
            str(row[category_column]): row for row in rows if row["cef"] == cef
        }
        values = [
            float(row_by_category[category]["category_error_rate"]) * RATE_SCALE
            if category in row_by_category
            else 0.0
            for category in categories
        ]
        lower_errors = [
            value
            - float(row_by_category[category]["category_error_rate_ci_lower"])
            * RATE_SCALE
            if category in row_by_category
            else 0.0
            for value, category in zip(values, categories)
        ]
        upper_errors = [
            float(row_by_category[category]["category_error_rate_ci_upper"])
            * RATE_SCALE
            - value
            if category in row_by_category
            else 0.0
            for value, category in zip(values, categories)
        ]
        positions = [position + offset for position in x_positions]
        plt.errorbar(
            positions,
            values,
            yerr=[lower_errors, upper_errors],
            fmt="o",
            capsize=4,
            linestyle="none",
            label=cef,
            color=CEF_COLORS[cef],
        )

    plt.xlabel(x_label, fontsize=LABEL_FONT_SIZE)
    plt.ylabel("Category error rate (%)", fontsize=LABEL_FONT_SIZE)
    plt.ylim(0, Y_AXIS_MAX * RATE_SCALE)
    plt.xticks(
        x_positions,
        categories,
        rotation=45,
        ha="right",
        fontsize=X_TICK_FONT_SIZE,
    )
    plt.tick_params(axis="y", labelsize=Y_TICK_FONT_SIZE)
    plt.grid(axis="y", alpha=0.3)
    plt.legend(title="CEF", fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()


def main() -> None:
    for speech_key, speech_config in SPEECH_TYPES.items():
        output_dir = output_dir_for(speech_key)
        for asr_key, asr_config in ASR_SYSTEMS.items():
            for error_key, error_config in ERROR_TYPES.items():
                rows = build_rate_rows(speech_key, asr_key, error_key)
                output_stem = f"{asr_key}_{error_key}_source_category_error_rates"
                write_csv(rows, output_dir / f"{output_stem}.csv")
                write_grouped_rate_plot(
                    rows,
                    output_dir / f"{output_stem}.png",
                    (
                        f"{asr_config['label']} {error_config['title_label']} "
                        f"rates by CEF ({speech_config['label']})"
                    ),
                    "source_pos",
                    x_label="Source POS tag",
                )
                print(f"Wrote CEF source category error rates to {output_dir / f'{output_stem}.png'}")

                group_rows = read_source_group_rate_rows(
                    input_file_for(
                        speech_key,
                        asr_key,
                        f"{error_key}_source_group_error_rates",
                    )
                )
                group_output_stem = f"{asr_key}_{error_key}_source_group_error_rates"
                write_csv(group_rows, output_dir / f"{group_output_stem}.csv")
                write_grouped_rate_plot(
                    group_rows,
                    output_dir / f"{group_output_stem}.png",
                    (
                        f"{asr_config['label']} {error_config['title_label']} "
                        f"group rates by CEF ({speech_config['label']})"
                    ),
                    "pos_group",
                    x_label="Source aggregated word category",
                    order=POS_GROUP_ORDER,
                    figure_size=(8, 6),
                )
                print(f"Wrote CEF source group error rates to {output_dir / f'{group_output_stem}.png'}")


if __name__ == "__main__":
    main()
