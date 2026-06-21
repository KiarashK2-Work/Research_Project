from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results" / "word_cat"
INPUT_DIR = RESULTS_DIR / "insertion_category_counts"
GROUP_ORDER = ["open_class", "closed_class", "other"]
DESTINATION_POS_FIELD = "destination_pos"
DESTINATION_GROUP_FIELD = "destination_group"
Y_AXIS_MAX = 900
PERCENTAGE_Y_AXIS_MAX = 65
LABEL_FONT_SIZE = 20
TICK_FONT_SIZE = 20
LEGEND_FONT_SIZE = 20

SPEECH_TYPES = {
    "read": {
        "label": "read",
    },
    "hmi": {
        "label": "HMI",
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


def iter_insertion_tables():
    yield from sorted(INPUT_DIR.glob("*_insertion_destination_table.csv"))


def format_percentage(count: int, total: int) -> str:
    if total == 0:
        return "0.0000"
    return f"{count / total * 100:.4f}"


def read_grouped_rows(input_file: Path) -> list[dict[str, str | int]]:
    aggregated: dict[tuple[str, str, str], dict[str, int]] = {}
    totals: dict[tuple[str, str], int] = {}

    with input_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {
            "model",
            "speech_type",
            DESTINATION_POS_FIELD,
            "count",
            "total_insertions",
        }
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{input_file} is missing required column(s): {missing}")

        for row_number, row in enumerate(reader, start=2):
            destination_pos = row[DESTINATION_POS_FIELD]
            try:
                destination_group = POS_GROUPS[destination_pos]
            except KeyError as error:
                raise ValueError(
                    f"No aggregate group configured for POS tag "
                    f"{destination_pos!r} in {input_file}"
                ) from error

            try:
                count = int(row["count"])
                total_insertions = int(row["total_insertions"])
            except ValueError as error:
                raise ValueError(
                    f"Invalid count value at {input_file}:{row_number}"
                ) from error

            model = row["model"]
            speech_type = row["speech_type"]
            key = (model, speech_type, destination_group)
            total_key = (model, speech_type)

            aggregated.setdefault(key, {"count": 0})["count"] += count
            existing_total = totals.get(total_key)
            if existing_total is not None and existing_total != total_insertions:
                raise ValueError(
                    f"Inconsistent total_insertions for {model}/{speech_type} "
                    f"in {input_file}"
                )
            totals[total_key] = total_insertions

    rows: list[dict[str, str | int]] = []
    model_speech_pairs = sorted(totals, key=lambda item: (item[0], item[1]))
    for model, speech_type in model_speech_pairs:
        total_insertions = totals[(model, speech_type)]
        for destination_group in GROUP_ORDER:
            count = aggregated.get((model, speech_type, destination_group), {}).get(
                "count", 0
            )
            rows.append(
                {
                    "model": model,
                    "speech_type": speech_type,
                    DESTINATION_GROUP_FIELD: destination_group,
                    "count": count,
                    "percentage": format_percentage(count, total_insertions),
                    "total_insertions": total_insertions,
                }
            )

    return rows


def write_table(rows: list[dict[str, str | int]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "speech_type",
        DESTINATION_GROUP_FIELD,
        "count",
        "percentage",
        "total_insertions",
    ]

    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rows_to_values(
    rows: list[dict[str, str | int]], value_field: str
) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = {}

    for row in rows:
        speech_type = str(row["speech_type"])
        destination_group = str(row[DESTINATION_GROUP_FIELD])
        values.setdefault(speech_type, {})[destination_group] = float(row[value_field])

    return values


def write_grouped_plot(
    rows: list[dict[str, str | int]],
    output_file: Path,
    title: str,
    y_label: str,
    value_field: str,
    y_axis_max: float,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    values_by_speech = rows_to_values(rows, value_field)

    bar_width = 0.38
    x_positions = list(range(len(GROUP_ORDER)))

    plt.figure(figsize=(8, 6))
    for index, speech_config in enumerate(SPEECH_TYPES.values()):
        speech_label = speech_config["label"]
        offset = (index - 0.5) * bar_width
        values = [
            values_by_speech.get(speech_label, {}).get(destination_group, 0.0)
            for destination_group in GROUP_ORDER
        ]
        positions = [position + offset for position in x_positions]
        plt.bar(positions, values, width=bar_width, label=speech_label)

    plt.xlabel("Destination aggregated word category", fontsize=LABEL_FONT_SIZE)
    plt.ylabel(y_label, fontsize=LABEL_FONT_SIZE)
    plt.ylim(0, y_axis_max)
    plt.xticks(x_positions, GROUP_ORDER, rotation=20, ha="right", fontsize=TICK_FONT_SIZE)
    plt.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
    plt.grid(axis="y", alpha=0.3)
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()


def model_label(model: str) -> str:
    if model == "chirp":
        return "Chirp"
    if model == "whisper":
        return "Whisper"
    return model


def main() -> None:
    input_files = list(iter_insertion_tables())
    if not input_files:
        raise FileNotFoundError(f"No insertion destination tables found in {INPUT_DIR}")

    for input_file in input_files:
        rows = read_grouped_rows(input_file)
        model = str(rows[0]["model"]) if rows else input_file.name.split("_", 1)[0]
        output_stem = input_file.stem.replace(
            "_insertion_destination_table",
            "_insertion_destination_group_table",
        )
        table_file = input_file.with_name(f"{output_stem}.csv")
        counts_plot = input_file.with_name(
            f"{model}_insertion_destination_group_counts.png"
        )
        percentages_plot = input_file.with_name(
            f"{model}_insertion_destination_group_percentages.png"
        )

        write_table(rows, table_file)
        write_grouped_plot(
            rows,
            counts_plot,
            f"{model_label(model)} insertion destination group counts",
            "Insertion count",
            "count",
            Y_AXIS_MAX,
        )
        write_grouped_plot(
            rows,
            percentages_plot,
            f"{model_label(model)} insertion destination group percentages",
            "Percentage of total insertions",
            "percentage",
            PERCENTAGE_Y_AXIS_MAX,
        )

        print(f"Wrote insertion destination group table to {table_file}")
        print(f"Wrote insertion destination group count plot to {counts_plot}")
        print(
            "Wrote insertion destination group percentage plot to "
            f"{percentages_plot}"
        )


if __name__ == "__main__":
    main()
