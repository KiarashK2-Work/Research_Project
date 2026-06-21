from __future__ import annotations

import csv
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL = "whisper"
SPEECH = "hmi"
RESULT_FILE = (
    SCRIPT_DIR
    / "data"
    / MODEL
    / f"nl_{SPEECH}_g4"
    / "score_wer"
    / "result.txt"
)
OUTPUT_FILE = (
    SCRIPT_DIR
    / "results"
    / "utt_length"
    / SPEECH
    / MODEL
    / f"utt_length_{SPEECH}_{MODEL}.csv"
)

ID_RE = re.compile(r"^id:\s+\((?P<utterance_id>[^)]+)\)")
USER_ID_RE = re.compile(r"(?:^|-)(?P<user_id>n\d+)(?:-|$)")
SCORES_RE = re.compile(
    r"^Scores:\s+\(#C #S #D #I\)\s+"
    r"(?P<correct>\d+)\s+"
    r"(?P<substitutions>\d+)\s+"
    r"(?P<deletions>\d+)\s+"
    r"(?P<insertions>\d+)"
)


def extract_user_id(utterance_id: str) -> str:
    match = USER_ID_RE.search(utterance_id)
    return match.group("user_id") if match else ""


def parse_result_file(result_file: Path) -> list[dict[str, int | str | float]]:
    rows: list[dict[str, int | str | float]] = []
    current_utterance_id: str | None = None

    with result_file.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            id_match = ID_RE.match(line.strip())
            if id_match:
                current_utterance_id = id_match.group("utterance_id")
                continue

            scores_match = SCORES_RE.match(line.strip())
            if not scores_match or current_utterance_id is None:
                continue

            correct = int(scores_match.group("correct"))
            substitutions = int(scores_match.group("substitutions"))
            deletions = int(scores_match.group("deletions"))
            insertions = int(scores_match.group("insertions"))

            utterance_length = correct + substitutions + deletions
            errors = substitutions + deletions + insertions
            wer = errors / utterance_length if utterance_length else 0.0
            substitution_rate = substitutions / utterance_length if utterance_length else 0.0
            deletion_rate = deletions / utterance_length if utterance_length else 0.0
            insertion_rate = insertions / utterance_length if utterance_length else 0.0

            rows.append(
                {
                    "utterance_id": current_utterance_id,
                    "user_id": extract_user_id(current_utterance_id),
                    "utterance_length": utterance_length,
                    "substitutions": substitutions,
                    "deletions": deletions,
                    "insertions": insertions,
                    "wer": round(wer, 6),
                    "insertion_rate": round(insertion_rate, 6),
                    "deletion_rate": round(deletion_rate, 6),
                    "substitution_rate": round(substitution_rate, 6),
                }
            )
            current_utterance_id = None

    return rows


def write_csv(rows: list[dict[str, int | str | float]], output_file: Path) -> None:
    fieldnames = [
        "utterance_id",
        "user_id",
        "utterance_length",
        "substitutions",
        "deletions",
        "insertions",
        "wer",
        "insertion_rate",
        "deletion_rate",
        "substitution_rate",
    ]

    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not RESULT_FILE.exists():
        raise FileNotFoundError(f"Could not find result file: {RESULT_FILE}")

    rows = parse_result_file(RESULT_FILE)
    write_csv(rows, OUTPUT_FILE)
    print(f"Wrote {len(rows)} utterances to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
