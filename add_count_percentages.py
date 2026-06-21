from __future__ import annotations

import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results" / "word_cat"
PERCENTAGE_FIELD = "percentage"


def iter_count_csv_files():
    yield from sorted(RESULTS_DIR.rglob("*_counts_*.csv"))


def get_tag_field(fieldnames: list[str]) -> str:
    for fieldname in fieldnames:
        if fieldname not in {"count", PERCENTAGE_FIELD}:
            return fieldname

    raise ValueError("Could not find a tag field in CSV header.")


def format_percentage(count: int, total: int) -> str:
    if total == 0:
        return "0.0000"
    return f"{count / total * 100:.4f}"


def add_percentages(csv_file: Path) -> None:
    with csv_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {csv_file}")

        original_fieldnames = reader.fieldnames
        tag_field = get_tag_field(original_fieldnames)
        rows = list(reader)

    total_row = next((row for row in rows if row[tag_field] == "TOTAL"), None)
    if total_row is not None:
        total = int(total_row["count"])
    else:
        total = sum(int(row["count"]) for row in rows)

    for row in rows:
        count = int(row["count"])
        if row[tag_field] == "TOTAL":
            row[PERCENTAGE_FIELD] = "100.0000"
        else:
            row[PERCENTAGE_FIELD] = format_percentage(count, total)

    fieldnames = [
        fieldname
        for fieldname in original_fieldnames
        if fieldname != PERCENTAGE_FIELD
    ]
    fieldnames.append(PERCENTAGE_FIELD)

    with csv_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    csv_files = list(iter_count_csv_files())
    if not csv_files:
        raise FileNotFoundError(f"No count CSV files found under {RESULTS_DIR}")

    for csv_file in csv_files:
        add_percentages(csv_file)
        print(f"Added percentages to {csv_file}")


if __name__ == "__main__":
    main()
