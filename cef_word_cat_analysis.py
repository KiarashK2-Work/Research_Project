from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import spacy
from spacy.tokens import Doc


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = SCRIPT_DIR / "data"
DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "results" / "word_cat"
DEFAULT_SPEAKERS_FILE = SCRIPT_DIR / "data" / "speakers.txt"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "results" / "CEF" / "word_cat"

SPACY_MODEL = "nl_core_news_lg"
BATCH_SIZE = 1000
N_PROCESS = 1
CONFIDENCE_Z = 1.96

CEF_LEVELS = ("A1", "A2", "B1")
SPEECH_TYPES = {
    "read": {"data_dir": "nl_read_g4", "label": "read"},
    "hmi": {"data_dir": "nl_hmi_g4", "label": "HMI"},
}
MODELS = {
    "google": {"data_dir": "google", "label": "chirp", "title": "Chirp"},
    "whisper": {"data_dir": "whisper", "label": "whisper", "title": "Whisper"},
}
POS_GROUP_ORDER = ("open_class", "closed_class", "other")
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
CEF_COLORS = {"A1": "#1f77b4", "A2": "#ff7f0e", "B1": "#2ca02c"}
ID_RE = re.compile(r"\((?P<utterance_id>[^)]+)\)\s*$")
USER_ID_RE = re.compile(r"(?P<user_id>n\d+)", re.IGNORECASE)


def extract_user_id(utterance_id: str) -> str:
    match = USER_ID_RE.search(utterance_id)
    return match.group("user_id").lower() if match else ""


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


def parse_trn_file(trn_file: Path) -> list[dict[str, str | list[str]]]:
    rows: list[dict[str, str | list[str]]] = []
    with trn_file.open(encoding="utf-8", errors="replace") as handle:
        for row_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            match = ID_RE.search(stripped)
            if not match:
                raise ValueError(f"Could not parse utterance id at {trn_file}:{row_number}")

            utterance_id = match.group("utterance_id")
            text = ID_RE.sub("", stripped).strip()
            rows.append(
                {
                    "utterance_id": utterance_id,
                    "user_id": extract_user_id(utterance_id),
                    "tokens": text.split() if text else [],
                }
            )

    return rows


def make_docs(nlp, token_lists: list[list[str]]):
    for tokens in token_lists:
        yield Doc(nlp.vocab, words=[token.lower() for token in tokens])


def count_trn_categories_by_cef(
    trn_file: Path,
    cef_by_speaker: dict[str, str],
    nlp,
) -> tuple[dict[str, Counter[str]], dict[str, Counter[str]], int]:
    trn_rows = parse_trn_file(trn_file)
    filtered_rows = [
        row for row in trn_rows if cef_by_speaker.get(str(row["user_id"])) in CEF_LEVELS
    ]
    token_lists = [row["tokens"] for row in filtered_rows]  # type: ignore[list-item]
    docs = nlp.pipe(make_docs(nlp, token_lists), batch_size=BATCH_SIZE, n_process=N_PROCESS)

    pos_counts = {cef: Counter() for cef in CEF_LEVELS}
    dep_counts = {cef: Counter() for cef in CEF_LEVELS}
    utterance_count = 0
    for row, doc in zip(filtered_rows, docs):
        cef = cef_by_speaker[str(row["user_id"])]
        pos_counts[cef].update(token.pos_ for token in doc)
        dep_counts[cef].update(token.dep_ for token in doc)
        utterance_count += 1

    return pos_counts, dep_counts, utterance_count


def pos_group_counts(pos_counts: dict[str, Counter[str]]) -> dict[str, Counter[str]]:
    grouped = {cef: Counter() for cef in CEF_LEVELS}
    for cef, counts in pos_counts.items():
        for pos_tag, count in counts.items():
            group = POS_GROUPS.get(pos_tag)
            if group is None:
                raise ValueError(f"No POS group configured for POS tag {pos_tag!r}")
            grouped[cef][group] += count
    return grouped


def add_percentages(
    count_rows: list[dict[str, str | int | float]], category_column: str
) -> list[dict[str, str | int | float]]:
    totals = Counter()
    for row in count_rows:
        totals[str(row["cef"])] += int(row["count"])

    rows: list[dict[str, str | int | float]] = []
    for row in count_rows:
        total = totals[str(row["cef"])]
        count = int(row["count"])
        new_row = dict(row)
        new_row["percentage"] = round((count / total) * 100, 4) if total else 0.0
        rows.append(new_row)

    return rows


def counts_to_rows(
    counts_by_cef: dict[str, Counter[str]],
    category_column: str,
    speech: str,
    model: str,
    order: Iterable[str] | None = None,
) -> list[dict[str, str | int | float]]:
    categories = list(order) if order is not None else sorted(
        {category for counts in counts_by_cef.values() for category in counts}
    )
    rows: list[dict[str, str | int | float]] = []
    for cef in CEF_LEVELS:
        for category in categories:
            count = counts_by_cef[cef].get(category, 0)
            rows.append(
                {
                    "speech": speech,
                    "model": MODELS[model]["label"],
                    "cef": cef,
                    category_column: category,
                    "count": count,
                }
            )
    return add_percentages(rows, category_column)


def read_error_rows(
    error_file: Path, cef_by_speaker: dict[str, str], speech: str, model: str
) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    with error_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {
            "utterance_id",
            "token_index",
            "error_type",
            "source_word",
            "source_pos",
            "source_dep",
            "destination_word",
            "destination_pos",
            "destination_dep",
        }
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{error_file} is missing required column(s): {missing}")

        for row in reader:
            user_id = extract_user_id(row["utterance_id"])
            cef = cef_by_speaker.get(user_id)
            if cef not in CEF_LEVELS:
                continue

            rows.append(
                {
                    "speech": speech,
                    "model": MODELS[model]["label"],
                    "cef": cef,
                    "utterance_id": row["utterance_id"],
                    "user_id": user_id,
                    "token_index": int(row["token_index"]),
                    "error_type": row["error_type"],
                    "source_word": row["source_word"],
                    "source_pos": row["source_pos"],
                    "source_pos_group": POS_GROUPS.get(row["source_pos"], row["source_pos"]),
                    "source_dep": row["source_dep"],
                    "destination_word": row["destination_word"],
                    "destination_pos": row["destination_pos"],
                    "destination_pos_group": POS_GROUPS.get(row["destination_pos"], row["destination_pos"]),
                    "destination_dep": row["destination_dep"],
                }
            )

    return rows


def count_error_categories(
    error_rows: list[dict[str, str | int]],
    error_type: str,
    category_column: str,
    speech: str,
    model: str,
    order: Iterable[str] | None = None,
) -> list[dict[str, str | int | float]]:
    counts = {cef: Counter() for cef in CEF_LEVELS}
    for row in error_rows:
        if row["error_type"] != error_type:
            continue
        category = str(row[category_column])
        if category in {"<del>", "<ins>", ""}:
            continue
        counts[str(row["cef"])][category] += 1

    output_column = category_column.replace("_group", "_category")
    return counts_to_rows(counts, output_column, speech, model, order)


def proportion_ci(successes: int, total: int) -> tuple[float, float, float]:
    if total == 0:
        return 0.0, 0.0, 0.0

    proportion = successes / total
    standard_error = math.sqrt(proportion * (1.0 - proportion) / total)
    margin = CONFIDENCE_Z * standard_error
    return proportion, max(0.0, proportion - margin), min(1.0, proportion + margin)


def source_group_error_rate_rows(
    error_rows: list[dict[str, str | int]],
    ref_group_counts: dict[str, Counter[str]],
    error_type: str,
    speech: str,
    model: str,
) -> list[dict[str, str | int | float]]:
    error_counts = {cef: Counter() for cef in CEF_LEVELS}
    for row in error_rows:
        if row["error_type"] != error_type:
            continue
        group = str(row["source_pos_group"])
        if group not in POS_GROUP_ORDER:
            continue
        error_counts[str(row["cef"])][group] += 1

    rows: list[dict[str, str | int | float]] = []
    for cef in CEF_LEVELS:
        for group in POS_GROUP_ORDER:
            errors = error_counts[cef].get(group, 0)
            references = ref_group_counts[cef].get(group, 0)
            rate, lower, upper = proportion_ci(errors, references)
            rows.append(
                {
                    "speech": speech,
                    "model": MODELS[model]["label"],
                    "cef": cef,
                    "error_type": error_type,
                    "pos_group": group,
                    "error_count": errors,
                    "reference_count": references,
                    "category_error_rate": round(rate, 6),
                    "category_error_rate_ci_lower": round(lower, 6),
                    "category_error_rate_ci_upper": round(upper, 6),
                }
            )
    return rows


def write_csv(rows: list[dict[str, str | int | float]], output_file: Path) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {output_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_grouped_bars(
    rows: list[dict[str, str | int | float]],
    output_file: Path,
    category_column: str,
    value_column: str,
    *,
    x_label: str,
    y_label: str,
    title: str,
    order: Iterable[str] | None = None,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    categories = list(order) if order is not None else sorted(
        {str(row[category_column]) for row in rows}
    )
    bar_width = 0.24
    x_positions = list(range(len(categories)))
    figure_width = max(9, len(categories) * 0.65)

    plt.figure(figsize=(figure_width, 6))
    for index, cef in enumerate(CEF_LEVELS):
        offset = (index - 1) * bar_width
        values_by_category = {
            str(row[category_column]): float(row[value_column])
            for row in rows
            if row["cef"] == cef
        }
        values = [values_by_category.get(category, 0.0) for category in categories]
        positions = [position + offset for position in x_positions]
        plt.bar(positions, values, width=bar_width, label=cef, color=CEF_COLORS[cef])

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(x_positions, categories, rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(title="CEF")
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()


def plot_source_rate_rows(
    rows: list[dict[str, str | int | float]],
    output_file: Path,
    *,
    title: str,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    x_positions = list(range(len(POS_GROUP_ORDER)))

    plt.figure(figsize=(8, 6))
    for index, cef in enumerate(CEF_LEVELS):
        offset = (index - 1) * 0.18
        row_by_group = {
            str(row["pos_group"]): row for row in rows if row["cef"] == cef
        }
        values = [float(row_by_group[group]["category_error_rate"]) for group in POS_GROUP_ORDER]
        lower_errors = [
            value - float(row_by_group[group]["category_error_rate_ci_lower"])
            for value, group in zip(values, POS_GROUP_ORDER)
        ]
        upper_errors = [
            float(row_by_group[group]["category_error_rate_ci_upper"]) - value
            for value, group in zip(values, POS_GROUP_ORDER)
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

    plt.xlabel("Source POS group")
    plt.ylabel("Category error rate")
    plt.ylim(bottom=0.0)
    plt.title(title)
    plt.xticks(x_positions, POS_GROUP_ORDER, rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(title="CEF")
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create CEF-split word-category analyses and plots."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--speakers-file", type=Path, default=DEFAULT_SPEAKERS_FILE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def process_dataset(
    speech: str,
    model: str,
    args: argparse.Namespace,
    cef_by_speaker: dict[str, str],
    nlp,
) -> tuple[list[dict[str, str | int | float]], list[dict[str, str | int | float]]]:
    model_label = MODELS[model]["label"]
    output_dir = args.output_root / speech / model_label
    prefix = f"cef_word_cat_{speech}_{model_label}"

    score_dir = args.data_root / MODELS[model]["data_dir"] / SPEECH_TYPES[speech]["data_dir"] / "score_wer"
    result_dir = args.results_root / speech / model
    ref_file = score_dir / "ref.trn"
    hyp_file = score_dir / "hyp.trn"
    error_file = result_dir / f"word_cat_errors_{speech}_{model}.csv"
    for input_file in (ref_file, hyp_file, error_file):
        if not input_file.exists():
            raise FileNotFoundError(f"Could not find input file: {input_file}")

    ref_pos_counts, ref_dep_counts, ref_utterances = count_trn_categories_by_cef(
        ref_file, cef_by_speaker, nlp
    )
    hyp_pos_counts, hyp_dep_counts, hyp_utterances = count_trn_categories_by_cef(
        hyp_file, cef_by_speaker, nlp
    )
    ref_group_counts = pos_group_counts(ref_pos_counts)
    hyp_group_counts = pos_group_counts(hyp_pos_counts)
    error_rows = read_error_rows(error_file, cef_by_speaker, speech, model)

    datasets = [
        ("ref_pos", ref_pos_counts, "pos_tag", None, "Reference POS tag"),
        ("hyp_pos", hyp_pos_counts, "pos_tag", None, "Hypothesis POS tag"),
        ("ref_dep", ref_dep_counts, "dep_tag", None, "Reference dependency tag"),
        ("hyp_dep", hyp_dep_counts, "dep_tag", None, "Hypothesis dependency tag"),
        ("ref_pos_group", ref_group_counts, "pos_group", POS_GROUP_ORDER, "Reference POS group"),
        ("hyp_pos_group", hyp_group_counts, "pos_group", POS_GROUP_ORDER, "Hypothesis POS group"),
    ]

    for stem, counts, category_column, order, label in datasets:
        rows = counts_to_rows(counts, category_column, speech, model, order)
        write_csv(rows, output_dir / f"{prefix}_{stem}_counts.csv")
        plot_grouped_bars(
            rows,
            output_dir / f"{prefix}_{stem}_counts.png",
            category_column,
            "count",
            x_label=label,
            y_label="Count",
            title=f"{MODELS[model]['title']} {label} counts by CEF ({SPEECH_TYPES[speech]['label']})",
            order=order,
        )
        plot_grouped_bars(
            rows,
            output_dir / f"{prefix}_{stem}_percentages.png",
            category_column,
            "percentage",
            x_label=label,
            y_label="Percentage",
            title=f"{MODELS[model]['title']} {label} distribution by CEF ({SPEECH_TYPES[speech]['label']})",
            order=order,
        )

    write_csv(error_rows, output_dir / f"{prefix}_errors.csv")

    error_specs = [
        ("deletion", "source_pos", "source_pos", None, "Deletion source POS"),
        ("deletion", "source_pos_group", "source_pos_category", POS_GROUP_ORDER, "Deletion source POS group"),
        ("substitution", "source_pos", "source_pos", None, "Substitution source POS"),
        ("substitution", "source_pos_group", "source_pos_category", POS_GROUP_ORDER, "Substitution source POS group"),
        ("substitution", "destination_pos", "destination_pos", None, "Substitution destination POS"),
        (
            "substitution",
            "destination_pos_group",
            "destination_pos_category",
            POS_GROUP_ORDER,
            "Substitution destination POS group",
        ),
        ("insertion", "destination_pos", "destination_pos", None, "Insertion destination POS"),
        (
            "insertion",
            "destination_pos_group",
            "destination_pos_category",
            POS_GROUP_ORDER,
            "Insertion destination POS group",
        ),
    ]
    for error_type, source_column, output_column, order, label in error_specs:
        rows = count_error_categories(error_rows, error_type, source_column, speech, model, order)
        stem = f"{prefix}_{error_type}_{source_column}_counts"
        write_csv(rows, output_dir / f"{stem}.csv")
        plot_grouped_bars(
            rows,
            output_dir / f"{stem}.png",
            output_column,
            "count",
            x_label=label,
            y_label="Count",
            title=f"{MODELS[model]['title']} {label} counts by CEF ({SPEECH_TYPES[speech]['label']})",
            order=order,
        )
        plot_grouped_bars(
            rows,
            output_dir / f"{stem}_percentages.png",
            output_column,
            "percentage",
            x_label=label,
            y_label="Percentage",
            title=f"{MODELS[model]['title']} {label} distribution by CEF ({SPEECH_TYPES[speech]['label']})",
            order=order,
        )

    rate_rows_all: list[dict[str, str | int | float]] = []
    for error_type in ("deletion", "substitution"):
        rows = source_group_error_rate_rows(error_rows, ref_group_counts, error_type, speech, model)
        rate_rows_all.extend(rows)
        write_csv(rows, output_dir / f"{prefix}_{error_type}_source_group_error_rates.csv")
        plot_source_rate_rows(
            rows,
            output_dir / f"{prefix}_{error_type}_source_group_error_rates.png",
            title=(
                f"{MODELS[model]['title']} {error_type} source POS-group error rates "
                f"by CEF ({SPEECH_TYPES[speech]['label']})"
            ),
        )

    print(
        f"{speech} {model_label}: ref_utterances={ref_utterances}, "
        f"hyp_utterances={hyp_utterances}, errors={len(error_rows)}"
    )
    return error_rows, rate_rows_all


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.results_root = args.results_root.resolve()
    args.speakers_file = args.speakers_file.resolve()
    args.output_root = args.output_root.resolve()

    cef_by_speaker = load_cef_by_speaker(args.speakers_file)
    try:
        nlp = spacy.load(SPACY_MODEL)
    except OSError as exc:
        raise SystemExit(
            f"Could not load spaCy model '{SPACY_MODEL}'. Install it with:\n"
            f"python -m spacy download {SPACY_MODEL}"
        ) from exc

    all_error_rows: list[dict[str, str | int | float]] = []
    all_rate_rows: list[dict[str, str | int | float]] = []
    for speech in SPEECH_TYPES:
        for model in MODELS:
            error_rows, rate_rows = process_dataset(
                speech, model, args, cef_by_speaker, nlp
            )
            all_error_rows.extend(error_rows)
            all_rate_rows.extend(rate_rows)

    write_csv(all_error_rows, args.output_root / "cef_word_cat_all_errors.csv")
    write_csv(all_rate_rows, args.output_root / "cef_word_cat_all_source_group_error_rates.csv")
    print(f"Wrote CEF word-category outputs to {args.output_root}")


if __name__ == "__main__":
    main()
