from __future__ import annotations

import csv
import re
from pathlib import Path

import spacy
from spacy.tokens import Doc


SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_FILE = SCRIPT_DIR / "data" / "whisper" / "nl_read_g4" / "score_wer" / "result.txt"
OUTPUT_FILE = SCRIPT_DIR / "results" / "word_cat" / "read" / "whisper" / "word_cat_errors_read_whisper.csv"
SPACY_MODEL = "nl_core_news_lg"
MAX_UTTERANCES = None
BATCH_SIZE = 1000
N_PROCESS = 1

ID_RE = re.compile(r"^id:\s+\((?P<utterance_id>[^)]+)\)")
LINE_RE = re.compile(r"^(?P<label>REF|HYP):\s+(?P<text>.*)$")


def is_placeholder(token: str) -> bool:
    return bool(token) and set(token) == {"*"}


def make_docs(nlp, token_lists: list[list[str]]):
    for tokens in token_lists:
        yield Doc(nlp.vocab, words=[token.lower() for token in tokens])


def parse_aligned_utterances(
    result_file: Path, max_utterances: int | None = None
) -> list[tuple[str, list[str], list[str]]]:
    utterances: list[tuple[str, list[str], list[str]]] = []
    current_utterance_id: str | None = None
    current_ref: list[str] | None = None
    current_hyp: list[str] | None = None
    utterances_seen = 0

    with result_file.open(encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")

            id_match = ID_RE.match(line.strip())
            if id_match:
                if max_utterances is not None and utterances_seen >= max_utterances:
                    break
                current_utterance_id = id_match.group("utterance_id")
                current_ref = None
                current_hyp = None
                continue

            line_match = LINE_RE.match(line)
            if not line_match:
                continue

            label = line_match.group("label")
            tokens = line_match.group("text").split()

            if label == "REF":
                current_ref = tokens
                continue

            current_hyp = tokens
            if current_utterance_id is None or current_ref is None:
                continue

            utterances_seen += 1

            if len(current_ref) != len(current_hyp):
                raise ValueError(
                    f"Aligned REF/HYP token count mismatch for {current_utterance_id}: "
                    f"{len(current_ref)} REF tokens, {len(current_hyp)} HYP tokens"
                )

            utterances.append((current_utterance_id, current_ref, current_hyp))
            current_ref = None
            current_hyp = None

    return utterances


def parse_error_rows(
    result_file: Path,
    nlp,
    max_utterances: int | None = None,
    batch_size: int = 1000,
    n_process: int = 1,
) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    utterances = parse_aligned_utterances(
        result_file, max_utterances=max_utterances
    )

    ref_token_lists = [
        [token for token in current_ref if not is_placeholder(token)]
        for _, current_ref, _ in utterances
    ]
    hyp_token_lists = [
        [token for token in current_hyp if not is_placeholder(token)]
        for _, _, current_hyp in utterances
    ]
    ref_docs = nlp.pipe(
        make_docs(nlp, ref_token_lists), batch_size=batch_size, n_process=n_process
    )
    hyp_docs = nlp.pipe(
        make_docs(nlp, hyp_token_lists), batch_size=batch_size, n_process=n_process
    )

    for (utterance_id, current_ref, current_hyp), ref_doc, hyp_doc in zip(
        utterances, ref_docs, hyp_docs
    ):
        ref_pos_iter = iter([token.pos_ for token in ref_doc])
        ref_dep_iter = iter([token.dep_ for token in ref_doc])
        hyp_pos_iter = iter([token.pos_ for token in hyp_doc])
        hyp_dep_iter = iter([token.dep_ for token in hyp_doc])

        aligned_ref_pos: list[str] = []
        aligned_ref_dep: list[str] = []
        aligned_hyp_pos: list[str] = []
        aligned_hyp_dep: list[str] = []
        for ref_token, hyp_token in zip(current_ref, current_hyp):
            aligned_ref_pos.append(
                "<del>" if is_placeholder(ref_token) else next(ref_pos_iter)
            )
            aligned_ref_dep.append(
                "<del>" if is_placeholder(ref_token) else next(ref_dep_iter)
            )
            aligned_hyp_pos.append(
                "<ins>" if is_placeholder(hyp_token) else next(hyp_pos_iter)
            )
            aligned_hyp_dep.append(
                "<ins>" if is_placeholder(hyp_token) else next(hyp_dep_iter)
            )

        for token_index, (
            ref_token,
            hyp_token,
            ref_pos,
            ref_dep,
            hyp_pos,
            hyp_dep,
        ) in enumerate(
            zip(
                current_ref,
                current_hyp,
                aligned_ref_pos,
                aligned_ref_dep,
                aligned_hyp_pos,
                aligned_hyp_dep,
            )
        ):
            ref_is_placeholder = is_placeholder(ref_token)
            hyp_is_placeholder = is_placeholder(hyp_token)

            if ref_is_placeholder and hyp_is_placeholder:
                continue
            if ref_is_placeholder:
                error_type = "insertion"
                source_word = "<ins>"
                source_pos = "<ins>"
                source_dep = "<ins>"
                destination_word = hyp_token.lower()
                destination_pos = hyp_pos
                destination_dep = hyp_dep
            elif hyp_is_placeholder:
                error_type = "deletion"
                source_word = ref_token.lower()
                source_pos = ref_pos
                source_dep = ref_dep
                destination_word = "<del>"
                destination_pos = "<del>"
                destination_dep = "<del>"
            elif ref_token.lower() != hyp_token.lower():
                error_type = "substitution"
                source_word = ref_token.lower()
                source_pos = ref_pos
                source_dep = ref_dep
                destination_word = hyp_token.lower()
                destination_pos = hyp_pos
                destination_dep = hyp_dep
            else:
                continue

            rows.append(
                {
                    "utterance_id": utterance_id,
                    "token_index": token_index,
                    "error_type": error_type,
                    "source_word": source_word,
                    "source_pos": source_pos,
                    "source_dep": source_dep,
                    "destination_word": destination_word,
                    "destination_pos": destination_pos,
                    "destination_dep": destination_dep,
                }
            )

    return rows


def write_csv(rows: list[dict[str, str | int]], output_file: Path) -> None:
    fieldnames = [
        "utterance_id",
        "token_index",
        "error_type",
        "source_word",
        "source_pos",
        "source_dep",
        "destination_word",
        "destination_pos",
        "destination_dep",
    ]

    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not RESULT_FILE.exists():
        raise FileNotFoundError(f"Could not find result file: {RESULT_FILE}")

    try:
        nlp = spacy.load(SPACY_MODEL)
    except OSError as exc:
        raise SystemExit(
            f"Could not load spaCy model '{SPACY_MODEL}'. Install it with:\n"
            f"python -m spacy download {SPACY_MODEL}"
        ) from exc

    rows = parse_error_rows(
        RESULT_FILE,
        nlp,
        max_utterances=MAX_UTTERANCES,
        batch_size=BATCH_SIZE,
        n_process=N_PROCESS,
    )
    write_csv(rows, OUTPUT_FILE)
    print(f"Wrote {len(rows)} word-category errors to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
