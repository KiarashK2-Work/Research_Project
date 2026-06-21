from __future__ import annotations

import csv
import time
from collections import Counter
from pathlib import Path

import spacy
from spacy.tokens import Doc


SCRIPT_DIR = Path(__file__).resolve().parent
ASR_SYSTEMS = {
    "whisper": {
        "data_dir": "whisper",
        "result_dir": "whisper",
        "label": "Whisper",
    },
    "google": {
        "data_dir": "google",
        "result_dir": "google",
        "label": "Chirp",
    },
}
SPEECH_TYPES = {
    "read": {
        "data_dir": "nl_read_g4",
        "result_dir": "read",
        "label": "read",
    },
    "hmi": {
        "data_dir": "nl_hmi_g4",
        "result_dir": "hmi",
        "label": "HMI",
    },
}
SPACY_MODEL = "nl_core_news_lg"
MAX_ROWS = None
PRINT_WORDS = False
BATCH_SIZE = 1000
N_PROCESS = 1


def parse_trn_tokens(trn_file: Path, max_rows: int | None = None) -> list[list[str]]:
    utterances: list[list[str]] = []

    with trn_file.open(encoding="utf-8", errors="replace") as handle:
        for row_index, line in enumerate(handle):
            if max_rows is not None and row_index >= max_rows:
                break
            text = line.rsplit("\t", 1)[0].strip()
            if text:
                utterances.append(text.split())

    return utterances


def make_docs(nlp, utterances: list[list[str]]):
    for tokens in utterances:
        yield Doc(nlp.vocab, words=[token.lower() for token in tokens])


def count_pos_and_dep_tags(
    trn_file: Path,
    nlp,
    max_rows: int | None = None,
    print_words: bool = False,
    batch_size: int = 1000,
    n_process: int = 1,
) -> tuple[Counter[str], Counter[str]]:
    pos_counts: Counter[str] = Counter()
    dep_counts: Counter[str] = Counter()
    utterances = parse_trn_tokens(trn_file, max_rows=max_rows)
    docs = nlp.pipe(
        make_docs(nlp, utterances), batch_size=batch_size, n_process=n_process
    )

    for tokens, doc in zip(utterances, docs):
        pos_tags = [token.pos_ for token in doc]
        dep_tags = [token.dep_ for token in doc]
        pos_counts.update(pos_tags)
        dep_counts.update(dep_tags)

        if print_words:
            for token, pos_tag, dep_tag in zip(tokens, pos_tags, dep_tags):
                print(f"{token}\t{pos_tag}\t{dep_tag}")

    return pos_counts, dep_counts


def write_counts_csv(counts: Counter[str], output_file: Path, tag_field: str) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[tag_field, "count"])
        writer.writeheader()

        for tag, count in sorted(counts.items()):
            writer.writerow({tag_field: tag, "count": count})

        writer.writerow({tag_field: "TOTAL", "count": sum(counts.values())})


def iter_dataset_configs():
    for speech_key, speech_config in SPEECH_TYPES.items():
        for asr_key, asr_config in ASR_SYSTEMS.items():
            score_dir = (
                SCRIPT_DIR
                / "data"
                / asr_config["data_dir"]
                / speech_config["data_dir"]
                / "score_wer"
            )
            output_dir = (
                SCRIPT_DIR
                / "results"
                / "word_cat"
                / speech_config["result_dir"]
                / asr_config["result_dir"]
            )

            yield {
                "speech_key": speech_key,
                "speech_label": speech_config["label"],
                "asr_key": asr_key,
                "asr_label": asr_config["label"],
                "ref_file": score_dir / "ref.trn",
                "hyp_file": score_dir / "hyp.trn",
                "ref_pos_output": output_dir
                / f"ref_pos_counts_{speech_key}_{asr_key}.csv",
                "ref_dep_output": output_dir
                / f"ref_dep_counts_{speech_key}_{asr_key}.csv",
                "hyp_pos_output": output_dir
                / f"hyp_pos_counts_{speech_key}_{asr_key}.csv",
                "hyp_dep_output": output_dir
                / f"hyp_dep_counts_{speech_key}_{asr_key}.csv",
            }


def process_dataset(config: dict[str, str | Path], nlp) -> float:
    ref_file = config["ref_file"]
    hyp_file = config["hyp_file"]

    if not isinstance(ref_file, Path) or not isinstance(hyp_file, Path):
        raise TypeError("Dataset input paths must be Path objects.")

    if not ref_file.exists():
        raise FileNotFoundError(f"Could not find ref.trn file: {ref_file}")
    if not hyp_file.exists():
        raise FileNotFoundError(f"Could not find hyp.trn file: {hyp_file}")

    processing_start = time.perf_counter()
    pos_counts, dep_counts = count_pos_and_dep_tags(
        ref_file,
        nlp,
        max_rows=MAX_ROWS,
        print_words=PRINT_WORDS,
        batch_size=BATCH_SIZE,
        n_process=N_PROCESS,
    )

    hyp_pos_counts, hyp_dep_counts = count_pos_and_dep_tags(
        hyp_file,
        nlp,
        max_rows=MAX_ROWS,
        print_words=PRINT_WORDS,
        batch_size=BATCH_SIZE,
        n_process=N_PROCESS,
    )
    processing_seconds = time.perf_counter() - processing_start

    write_counts_csv(pos_counts, config["ref_pos_output"], "pos_tag")
    write_counts_csv(dep_counts, config["ref_dep_output"], "dep_tag")
    write_counts_csv(hyp_pos_counts, config["hyp_pos_output"], "pos_tag")
    write_counts_csv(hyp_dep_counts, config["hyp_dep_output"], "dep_tag")

    print(
        f"{config['asr_label']} {config['speech_label']}: wrote "
        f"{sum(pos_counts.values())} reference POS tags to {config['ref_pos_output']}"
    )
    print(
        f"{config['asr_label']} {config['speech_label']}: wrote "
        f"{sum(dep_counts.values())} reference DEP tags to {config['ref_dep_output']}"
    )
    print(
        f"{config['asr_label']} {config['speech_label']}: wrote "
        f"{sum(hyp_pos_counts.values())} hypothesis POS tags to {config['hyp_pos_output']}"
    )
    print(
        f"{config['asr_label']} {config['speech_label']}: wrote "
        f"{sum(hyp_dep_counts.values())} hypothesis DEP tags to {config['hyp_dep_output']}"
    )
    print(
        f"{config['asr_label']} {config['speech_label']}: "
        f"processing time {processing_seconds:.3f} seconds"
    )

    return processing_seconds


def main() -> None:
    total_start = time.perf_counter()

    try:
        nlp = spacy.load(SPACY_MODEL)
    except OSError as exc:
        raise SystemExit(
            f"Could not load spaCy model '{SPACY_MODEL}'. Install it with:\n"
            f"python -m spacy download {SPACY_MODEL}"
        ) from exc

    processing_seconds = 0.0
    for config in iter_dataset_configs():
        processing_seconds += process_dataset(config, nlp)

    total_seconds = time.perf_counter() - total_start

    print(f"Processing time: {processing_seconds:.3f} seconds")
    print(f"Total runtime: {total_seconds:.3f} seconds")


if __name__ == "__main__":
    main()
