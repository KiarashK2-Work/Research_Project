from __future__ import annotations

import csv
import math
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results" / "word_cat"
SOURCE_RATE_DIR = RESULTS_DIR / "source_category_error_rates"
INSERTION_DIR = RESULTS_DIR / "insertion_category_counts"
OUTPUT_FILE = RESULTS_DIR / "word_category_open_closed_chi_square_tests.csv"

MODELS = ("chirp", "whisper")
SPEECH_TYPES = ("read", "HMI")
GROUPS_TO_COMPARE = ("open_class", "closed_class")
ALPHA = 0.05


def significance_code(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < ALPHA:
        return "*"
    return "ns"


def chi_square_2x2(
    group_a_success: int,
    group_a_total: int,
    group_b_success: int,
    group_b_total: int,
) -> tuple[float, float]:
    group_a_failure = group_a_total - group_a_success
    group_b_failure = group_b_total - group_b_success
    if min(group_a_success, group_a_failure, group_b_success, group_b_failure) < 0:
        raise ValueError("Success counts cannot exceed total counts")

    observed = [
        [group_a_success, group_a_failure],
        [group_b_success, group_b_failure],
    ]
    row_totals = [sum(row) for row in observed]
    column_totals = [
        observed[0][0] + observed[1][0],
        observed[0][1] + observed[1][1],
    ]
    grand_total = sum(row_totals)
    if grand_total == 0 or 0 in row_totals or 0 in column_totals:
        return 0.0, 1.0

    chi_square = 0.0
    for row_index, row in enumerate(observed):
        for column_index, value in enumerate(row):
            expected = row_totals[row_index] * column_totals[column_index] / grand_total
            chi_square += (value - expected) ** 2 / expected

    # For df=1, the chi-square survival function is erfc(sqrt(x / 2)).
    p_value = math.erfc(math.sqrt(chi_square / 2.0))
    return chi_square, p_value


def chi_square_50_50(open_count: int, closed_count: int) -> tuple[float, float]:
    total = open_count + closed_count
    if total == 0:
        return 0.0, 1.0

    expected = total / 2
    chi_square = ((open_count - expected) ** 2 / expected) + (
        (closed_count - expected) ** 2 / expected
    )
    p_value = math.erfc(math.sqrt(chi_square / 2.0))
    return chi_square, p_value


def read_source_group_rows(model: str, error_type: str) -> list[dict[str, str]]:
    input_file = SOURCE_RATE_DIR / f"{model}_{error_type}_source_group_error_rates.csv"
    with input_file.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_insertion_group_rows(model: str) -> list[dict[str, str]]:
    input_file = INSERTION_DIR / f"{model}_insertion_destination_group_table.csv"
    with input_file.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def rows_by_group(rows: list[dict[str, str]], speech_type: str, group_field: str):
    return {
        row[group_field]: row
        for row in rows
        if row["speech_type"] == speech_type and row[group_field] in GROUPS_TO_COMPARE
    }


def test_source_error_rate(model: str, speech_type: str, error_type: str) -> dict[str, str]:
    rows = rows_by_group(
        read_source_group_rows(model, error_type),
        speech_type,
        "pos_group",
    )
    open_row = rows["open_class"]
    closed_row = rows["closed_class"]
    return build_result_row(
        model=model,
        speech_type=speech_type,
        error_type=f"{error_type}_source",
        denominator_type="reference_count",
        open_count=int(open_row["error_count"]),
        open_total=int(open_row["reference_count"]),
        closed_count=int(closed_row["error_count"]),
        closed_total=int(closed_row["reference_count"]),
    )


def test_insertion_destination_rate(model: str, speech_type: str) -> dict[str, str]:
    rows = rows_by_group(
        read_insertion_group_rows(model),
        speech_type,
        "destination_group",
    )
    open_row = rows["open_class"]
    closed_row = rows["closed_class"]
    open_count = int(open_row["count"])
    closed_count = int(closed_row["count"])
    total = open_count + closed_count
    chi_square, p_value = chi_square_50_50(open_count, closed_count)
    return {
        "model": model,
        "speech_type": speech_type,
        "error_type": "insertion_destination",
        "comparison": "open_class_vs_closed_class",
        "test_type": "chi_square_goodness_of_fit_50_50",
        "denominator_type": "open_closed_insertions",
        "open_class_error_count": str(open_count),
        "open_class_total": str(total),
        "open_class_rate": f"{open_count / total:.6f}" if total else "0.000000",
        "closed_class_error_count": str(closed_count),
        "closed_class_total": str(total),
        "closed_class_rate": f"{closed_count / total:.6f}" if total else "0.000000",
        "rate_difference_open_minus_closed": (
            f"{(open_count - closed_count) / total:.6f}" if total else "0.000000"
        ),
        "chi_square": f"{chi_square:.6f}",
        "df": "1",
        "p_value": f"{p_value:.10g}",
        "significant": str(p_value < ALPHA),
        "significance": significance_code(p_value),
    }

def build_result_row(
    model: str,
    speech_type: str,
    error_type: str,
    denominator_type: str,
    open_count: int,
    open_total: int,
    closed_count: int,
    closed_total: int,
) -> dict[str, str]:
    chi_square, p_value = chi_square_2x2(
        open_count,
        open_total,
        closed_count,
        closed_total,
    )
    open_rate = open_count / open_total if open_total else 0.0
    closed_rate = closed_count / closed_total if closed_total else 0.0
    return {
        "model": model,
        "speech_type": speech_type,
        "error_type": error_type,
        "comparison": "open_class_vs_closed_class",
        "test_type": "chi_square_independence_2x2",
        "denominator_type": denominator_type,
        "open_class_error_count": str(open_count),
        "open_class_total": str(open_total),
        "open_class_rate": f"{open_rate:.6f}",
        "closed_class_error_count": str(closed_count),
        "closed_class_total": str(closed_total),
        "closed_class_rate": f"{closed_rate:.6f}",
        "rate_difference_open_minus_closed": f"{open_rate - closed_rate:.6f}",
        "chi_square": f"{chi_square:.6f}",
        "df": "1",
        "p_value": f"{p_value:.10g}",
        "significant": str(p_value < ALPHA),
        "significance": significance_code(p_value),
    }


def write_results(rows: list[dict[str, str]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "speech_type",
        "error_type",
        "comparison",
        "test_type",
        "denominator_type",
        "open_class_error_count",
        "open_class_total",
        "open_class_rate",
        "closed_class_error_count",
        "closed_class_total",
        "closed_class_rate",
        "rate_difference_open_minus_closed",
        "chi_square",
        "df",
        "p_value",
        "significant",
        "significance",
    ]
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows: list[dict[str, str]] = []
    for model in MODELS:
        for speech_type in SPEECH_TYPES:
            rows.append(test_source_error_rate(model, speech_type, "deletion"))
            rows.append(test_source_error_rate(model, speech_type, "substitution"))
            rows.append(test_insertion_destination_rate(model, speech_type))

    write_results(rows, OUTPUT_FILE)
    print(f"Wrote chi-square test results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
