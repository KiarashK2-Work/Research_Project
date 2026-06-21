from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
ASR_SYSTEMS = {
    "whisper": {
        "result_dir": "whisper",
        "label": "Whisper",
    },
    "google": {
        "result_dir": "google",
        "label": "Chirp",
    },
}
SPEECH_TYPES = {
    "read": {
        "result_dir": "read",
        "label": "read speech",
    },
    "hmi": {
        "result_dir": "hmi",
        "label": "HMI speech",
    },
}
SOURCES = {
    "ref": "reference",
    "hyp": "hypothesis",
}
TAG_TYPES = {
    "pos": {
        "tag_field": "pos_tag",
        "x_label": "POS tag",
        "title_label": "POS tags",
    },
    "dep": {
        "tag_field": "dep_tag",
        "x_label": "Dependency tag",
        "title_label": "dependency tags",
    },
}
LABEL_FONT_SIZE = 24
TICK_FONT_SIZE = 18
X_TICK_FONT_SIZE = 14


def read_counts_csv(csv_file: Path, tag_field: str) -> Counter[str]:
    counts: Counter[str] = Counter()

    with csv_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            tag = row[tag_field]
            if tag == "TOTAL":
                continue
            counts[tag] = int(row["count"])

    return counts


def write_distribution_plot(
    counts: Counter[str],
    output_file: Path,
    x_label: str,
    title: str,
    tag_order: list[str] | None = None,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if tag_order is None:
        tags = [tag for tag, _ in sorted(counts.items())]
    else:
        ordered_tags = [tag for tag in tag_order if tag in counts]
        extra_tags = sorted(tag for tag in counts if tag not in tag_order)
        tags = ordered_tags + extra_tags
    values = [counts[tag] for tag in tags]
    figure_width = max(10, len(tags) * 0.45)

    plt.figure(figsize=(figure_width, 6))
    plt.bar(tags, values)
    plt.xlabel(x_label, fontsize=LABEL_FONT_SIZE)
    plt.ylabel("Count", fontsize=LABEL_FONT_SIZE)
    plt.tick_params(axis="x", labelsize=X_TICK_FONT_SIZE)
    plt.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()


def iter_plot_configs():
    for speech_key, speech_config in SPEECH_TYPES.items():
        for asr_key, asr_config in ASR_SYSTEMS.items():
            results_dir = (
                SCRIPT_DIR
                / "results"
                / "word_cat"
                / speech_config["result_dir"]
                / asr_config["result_dir"]
            )
            for source_key, source_label in SOURCES.items():
                for tag_key, tag_config in TAG_TYPES.items():
                    csv_file = (
                        results_dir
                        / f"{source_key}_{tag_key}_counts_{speech_key}_{asr_key}.csv"
                    )
                    yield {
                        "csv_file": csv_file,
                        "speech_key": speech_key,
                        "asr_key": asr_key,
                        "source_key": source_key,
                        "tag_key": tag_key,
                        "tag_field": tag_config["tag_field"],
                        "x_label": tag_config["x_label"],
                        "title": (
                            f"Distribution of {source_label} "
                            f"{tag_config['title_label']} "
                            f"for {speech_config['label']}"
                        ),
                    }


def read_google_ref_pos_order() -> list[str]:
    csv_file = (
        SCRIPT_DIR
        / "results"
        / "word_cat"
        / SPEECH_TYPES["read"]["result_dir"]
        / ASR_SYSTEMS["google"]["result_dir"]
        / "ref_pos_counts_read_google.csv"
    )
    counts = read_counts_csv(csv_file, "pos_tag")
    return [
        tag
        for tag, _ in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]


def main() -> None:
    google_ref_pos_order = read_google_ref_pos_order()
    for plot_config in iter_plot_configs():
        csv_file = plot_config["csv_file"]
        tag_field = plot_config["tag_field"]
        output_file = csv_file.with_name(f"{csv_file.stem}_distribution.png")

        if not csv_file.exists():
            raise FileNotFoundError(f"Could not find counts CSV file: {csv_file}")

        counts = read_counts_csv(csv_file, tag_field)
        tag_order = None
        if (
            plot_config["asr_key"] == "google"
            and plot_config["source_key"] == "ref"
            and plot_config["tag_key"] == "pos"
            and plot_config["speech_key"] in {"read", "hmi"}
        ):
            tag_order = google_ref_pos_order

        write_distribution_plot(
            counts,
            output_file,
            plot_config["x_label"],
            plot_config["title"],
            tag_order,
        )
        print(f"Wrote distribution plot to {output_file}")


if __name__ == "__main__":
    main()
