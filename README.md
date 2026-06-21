# Linguistic sources of performance disparities in Dutch Automatic Speech Recognition for non-native adults

This repository contains the research project for my bachelor's thesis at TU
Delft.

**Author:** Kiarash Karimi <makarimi@tudelft.nl>

The project analyzes automatic speech recognition (ASR) errors in Dutch read
speech and human-machine interaction (HMI) speech. It compares Google Chirp and
Whisper outputs, with particular attention to:

- utterance length and word error rate (WER);
- substitutions, deletions, and insertions;
- word length;
- part-of-speech (POS) and dependency categories;
- open-class, closed-class, and other POS groups; and
- speaker proficiency at CEF levels A1, A2, and B1.

Most scripts read ASR scoring files from `data/`, then write CSV summaries and
PNG figures to `results/`. The repository contains analysis scripts rather than
a single application or command-line pipeline, so scripts are normally run
individually in the order required for a particular analysis.

## Repository Layout

```text
data/
  google/                  Chirp scoring outputs
  whisper/                 Whisper scoring outputs
  whisper_V3_long_audio/   Additional Whisper outputs
  speakers.txt             Speaker metadata, including CEF level
results/
  CEF/                     Proficiency-level analyses
  utt_length/              Utterance-length analyses
  word_cat/                POS and word-category analyses
  word_length/             Word-length analyses
*.py                       Main analysis and plotting scripts
```

The scripts generally expect ESPnet-style scoring files such as `result.txt`,
`ref.trn`, and `hyp.trn`. Many input and output paths are configured as
constants near the top of each script. Check these constants before running a
script for a different speech type or ASR system.

## Requirements

The Python analyses require Python 3.10 or newer and:

```powershell
pip install matplotlib spacy
python -m spacy download nl_core_news_lg
python -m spacy download nl_core_news_md
```

## Typical Workflow

1. Place ASR scoring output under the appropriate `data/<model>/<dataset>/`
   directory.
2. Extract utterance-level, POS, or word-level data with scripts such as
   `utterance_length.py`, `word_category.py`, or
   `ref_pos_counts_parallel.py`.
3. Produce per-dataset summaries with `utt_length_analysis.py`,
   `word_cat_analysis.py`, or `word_length_analysis.py`.
4. Combine results across models and speech types with the `aggregate_*` and
   `*_combined_*` scripts.
5. Run the CEF, statistical-test, and final plotting scripts as needed.

Run scripts from the repository root, for example:

```powershell
python ref_pos_counts_parallel.py
python utterance_length.py
python utt_length_analysis.py --help
```

Scripts that implement `argparse` support `--help`. Other scripts are
configured by editing their constants.

## Script Reference

### Data Extraction and Base Counts

- `utterance_length.py`: Parses an ASR `result.txt` file and writes
  utterance-level lengths, error counts, WER, and substitution/deletion/
  insertion rates.
- `word_category.py`: Parses aligned REF/HYP output, assigns Dutch spaCy POS
  tags, and writes one row per substitution, deletion, or insertion.
- `ref_pos_counts.py`: Counts POS tags in one reference transcription file; a
  smaller single-dataset version of the broader counting script.
- `ref_pos_counts_parallel.py`: Counts POS and dependency tags in reference
  and hypothesis transcriptions for both models and speech types using spaCy
  batching.
- `ref_word_lengths.py`: Counts reference-word lengths and creates a CSV and
  distribution plot.
- `word_length_errors.py`: Adds source and destination word lengths to the
  word-category error rows.

### Utterance-Length Analysis

- `utt_length_analysis.py`: Summarizes WER and individual error rates by
  utterance length, calculates confidence intervals, and creates tables and
  plots.
- `utt_length_distribution_stats.py`: Produces descriptive utterance-length
  statistics, percentiles, and proportions below configured length cutoffs
  across speech types and models.
- `utt_length_read_vs_hmi.py`: Compares mean WER by utterance length between
  read and HMI speech for one configured ASR model.
- `cef_utt_length_analysis.py`: Joins utterance results to speaker CEF levels
  and compares length distributions and error rates for A1, A2, and B1
  speakers.

### POS and Word-Category Analysis

- `word_cat_analysis.py`: Counts and plots POS categories for deletion
  sources, substitution sources and destinations, and insertion destinations.
- `word_cat_combined_errors.py`: Creates one grouped plot containing the four
  POS error distributions for a selected model and speech type.
- `word_cat_combined_errors_table.py`: Combines POS error counts for all
  speech-type/model combinations into one CSV.
- `add_count_percentages.py`: Adds a percentage column to generated POS count
  CSV files.
- `add_source_category_error_rates.py`: Divides source-category error counts
  by reference POS counts and adds category-specific error rates.
- `substitution_category_pairs.py`: Counts source-to-destination POS pairs for
  substitutions and calculates their percentages and source-category rates.
- `top_substitution_pos_pair_rates.py`: Selects the highest-rate substitution
  POS pairs, excluding source categories below the configured minimum size.
- `aggregate_pos_group_distributions.py`: Collapses POS counts into
  open-class, closed-class, and other groups.
- `aggregate_word_cat_errors_combined.py`: Collapses the combined POS error
  table into the three broad POS groups.
- `aggregate_substitution_group_pairs.py`: Collapses substitution POS pairs
  into source/destination POS-group pairs across all datasets.
- `aggregate_source_category_error_rates.py`: Calculates and plots deletion
  and substitution rates for the broad source POS groups.
- `aggregate_insertion_category_counts.py`: Collapses insertion destinations
  into broad POS groups and writes count and percentage plots.
- `chi_square_word_category_tests.py`: Runs chi-square tests comparing
  open-class and closed-class error or insertion counts and writes the test
  results to CSV.
- `cef_word_cat_analysis.py`: Rebuilds reference, hypothesis, and aligned error
  summaries by CEF level, including POS counts, POS-group counts, error
  distributions, rates, and plots.

### Plotting

- `plot_ref_pos_counts.py`: Plots reference and hypothesis POS/dependency count
  CSVs produced by `ref_pos_counts_parallel.py`.
- `plot_source_category_error_rates.py`: Plots POS-specific deletion and
  substitution source-category error rates for Chirp and Whisper.
- `plot_insertion_category_counts.py`: Combines insertion destination counts
  across read and HMI speech into model-level tables and plots.
- `plot_cef_source_category_error_rates.py`: Compares POS-specific and grouped
  deletion/substitution rates across CEF levels.
- `plot_cef_insertion_destinations.py`: Compares insertion destination POS and
  POS-group percentages across CEF levels.

### Word-Length Analysis

- `word_length_analysis.py`: Counts and plots source/destination word lengths
  for substitutions, deletions, and insertions.

## Outputs

Generated CSV and PNG files are stored under `results/`. Existing outputs are
included in the repository, so rerunning a script may overwrite files with the
same names. The empty `results/utt_structure/` directory appears reserved for
future utterance-structure analyses.
