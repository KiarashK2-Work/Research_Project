from __future__ import annotations

import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results" / "word_cat"
RATE_FIELD = "category_error_rate"


def iter_source_pos_error_files():
    yield from sorted(RESULTS_DIR.rglob("word_cat_errors_*_source_pos.csv"))


def find_ref_pos_counts_file(error_file: Path) -> Path:
    count_files = sorted(error_file.parent.glob("ref_pos_counts_*.csv"))
    if not count_files:
        raise FileNotFoundError(
            f"Could not find a ref_pos_counts CSV in {error_file.parent}"
        )
    if len(count_files) > 1:
        raise ValueError(
            f"Expected one ref_pos_counts CSV in {error_file.parent}, "
            f"found {len(count_files)}"
        )
    return count_files[0]


def read_ref_pos_denominators(counts_file: Path) -> dict[str, int]:
    denominators: dict[str, int] = {}

    with counts_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "pos_tag" not in reader.fieldnames:
            raise ValueError(f"Expected a pos_tag column in {counts_file}")

        for row in reader:
            pos_tag = row["pos_tag"]
            if pos_tag == "TOTAL":
                continue
            denominators[pos_tag] = int(row["count"])

    return denominators


def format_rate(error_count: int, category_total: int) -> str:
    if category_total == 0:
        return ""
    return f"{error_count / category_total:.6f}"


def add_error_rates(error_file: Path) -> None:
    counts_file = find_ref_pos_counts_file(error_file)
    denominators = read_ref_pos_denominators(counts_file)

    with error_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {error_file}")
        if "source_pos" not in reader.fieldnames:
            raise ValueError(f"Expected a source_pos column in {error_file}")
        if "count" not in reader.fieldnames:
            raise ValueError(f"Expected a count column in {error_file}")

        original_fieldnames = reader.fieldnames
        rows = list(reader)

    for row in rows:
        source_pos = row["source_pos"]
        denominator = denominators.get(source_pos)
        if denominator is None:
            row[RATE_FIELD] = ""
            continue

        row[RATE_FIELD] = format_rate(int(row["count"]), denominator)

    fieldnames = [
        fieldname for fieldname in original_fieldnames if fieldname != RATE_FIELD
    ]
    fieldnames.append(RATE_FIELD)

    with error_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    error_files = list(iter_source_pos_error_files())
    if not error_files:
        raise FileNotFoundError(f"No source POS error CSV files found under {RESULTS_DIR}")

    for error_file in error_files:
        add_error_rates(error_file)
        print(f"Added source category error rates to {error_file}")


if __name__ == "__main__":
    main()
