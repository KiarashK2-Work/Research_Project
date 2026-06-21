from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results" / "CEF" / "word_cat"
CEF_LEVELS = ("A1", "A2", "B1")
CEF_COLORS = {"A1": "#1f77b4", "A2": "#ff7f0e", "B1": "#2ca02c"}
POS_GROUP_ORDER = ("open_class", "closed_class", "other")
POS_PERCENTAGE_Y_AXIS_MAX = 25
GROUP_PERCENTAGE_Y_AXIS_MAX = 65
POS_LABEL_FONT_SIZE = 24
POS_X_TICK_FONT_SIZE = 14
POS_Y_TICK_FONT_SIZE = 20
POS_LEGEND_FONT_SIZE = 20
GROUP_LABEL_FONT_SIZE = 20
GROUP_TICK_FONT_SIZE = 20
GROUP_LEGEND_FONT_SIZE = 20

SPEECH_TYPES = {
    "read": {"label": "read"},
    "hmi": {"label": "HMI"},
}
ASR_SYSTEMS = {
    "chirp": {"label": "Chirp"},
    "whisper": {"label": "Whisper"},
}


def input_dir_for(speech_key: str, asr_key: str) -> Path:
    return RESULTS_DIR / speech_key / asr_key


def output_dir_for(speech_key: str) -> Path:
    return RESULTS_DIR / speech_key / "insertion_destinations"


def input_file_for(speech_key: str, asr_key: str, stem: str) -> Path:
    return input_dir_for(speech_key, asr_key) / f"cef_word_cat_{speech_key}_{asr_key}_{stem}.csv"


def read_percentage_rows(input_file: Path, category_column: str) -> list[dict[str, str]]:
    with input_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"cef", category_column, "count", "percentage"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{input_file} is missing required column(s): {missing}")

        return list(reader)


def write_csv(rows: list[dict[str, str]], output_file: Path) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {output_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_grouped_percentage_plot(
    rows: list[dict[str, str]],
    output_file: Path,
    title: str,
    category_column: str,
    *,
    x_label: str,
    y_axis_max: float,
    label_font_size: int,
    x_tick_font_size: int,
    y_tick_font_size: int,
    legend_font_size: int,
    order: tuple[str, ...] | None = None,
    figure_size: tuple[float, float] | None = None,
    rotation: int = 45,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    categories = list(order) if order is not None else sorted(
        {row[category_column] for row in rows}
    )
    if not categories:
        raise ValueError(f"No insertion destination percentages found for {title}")

    max_percentage = max(float(row["percentage"]) for row in rows)
    y_limit = max(y_axis_max, math.ceil((max_percentage * 1.1) / 5.0) * 5.0)

    bar_width = 0.24
    x_positions = list(range(len(categories)))
    figure_width = max(12, len(categories) * 0.7)

    plt.figure(figsize=figure_size or (figure_width, 6))
    for index, cef in enumerate(CEF_LEVELS):
        offset = (index - 1) * bar_width
        values_by_category = {
            row[category_column]: float(row["percentage"])
            for row in rows
            if row["cef"] == cef
        }
        values = [values_by_category.get(category, 0.0) for category in categories]
        positions = [position + offset for position in x_positions]
        plt.bar(
            positions,
            values,
            width=bar_width,
            label=cef,
            color=CEF_COLORS[cef],
        )

    plt.xlabel(x_label, fontsize=label_font_size)
    plt.ylabel("Percentage of total insertions", fontsize=label_font_size)
    plt.ylim(0, y_limit)
    plt.xticks(
        x_positions,
        categories,
        rotation=rotation,
        ha="right",
        fontsize=x_tick_font_size,
    )
    plt.tick_params(axis="y", labelsize=y_tick_font_size)
    plt.grid(axis="y", alpha=0.3)
    plt.legend(fontsize=legend_font_size)
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()


def main() -> None:
    for speech_key, speech_config in SPEECH_TYPES.items():
        output_dir = output_dir_for(speech_key)
        for asr_key, asr_config in ASR_SYSTEMS.items():
            rows = read_percentage_rows(
                input_file_for(
                    speech_key,
                    asr_key,
                    "insertion_destination_pos_counts",
                ),
                "destination_pos",
            )
            output_stem = f"{asr_key}_insertion_destination_percentages"
            write_csv(rows, output_dir / f"{output_stem}.csv")
            write_grouped_percentage_plot(
                rows,
                output_dir / f"{output_stem}.png",
                (
                    f"{asr_config['label']} insertion destination percentages "
                    f"by CEF ({speech_config['label']})"
                ),
                "destination_pos",
                x_label="Destination POS tag",
                y_axis_max=POS_PERCENTAGE_Y_AXIS_MAX,
                label_font_size=POS_LABEL_FONT_SIZE,
                x_tick_font_size=POS_X_TICK_FONT_SIZE,
                y_tick_font_size=POS_Y_TICK_FONT_SIZE,
                legend_font_size=POS_LEGEND_FONT_SIZE,
            )
            print(f"Wrote CEF insertion destination percentages to {output_dir / f'{output_stem}.png'}")

            group_rows = read_percentage_rows(
                input_file_for(
                    speech_key,
                    asr_key,
                    "insertion_destination_pos_group_counts",
                ),
                "destination_pos_category",
            )
            group_output_stem = f"{asr_key}_insertion_destination_group_percentages"
            write_csv(group_rows, output_dir / f"{group_output_stem}.csv")
            write_grouped_percentage_plot(
                group_rows,
                output_dir / f"{group_output_stem}.png",
                (
                    f"{asr_config['label']} insertion destination group percentages "
                    f"by CEF ({speech_config['label']})"
                ),
                "destination_pos_category",
                x_label="Destination aggregated word category",
                y_axis_max=GROUP_PERCENTAGE_Y_AXIS_MAX,
                label_font_size=GROUP_LABEL_FONT_SIZE,
                x_tick_font_size=GROUP_TICK_FONT_SIZE,
                y_tick_font_size=GROUP_TICK_FONT_SIZE,
                legend_font_size=GROUP_LEGEND_FONT_SIZE,
                order=POS_GROUP_ORDER,
                figure_size=(8, 6),
                rotation=20,
            )
            print(f"Wrote CEF insertion destination group percentages to {output_dir / f'{group_output_stem}.png'}")


if __name__ == "__main__":
    main()
