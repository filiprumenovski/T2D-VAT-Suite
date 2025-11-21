# T2D-VAT-Suite: Two-Tool Architecture for Proteomics Analysis

A modular Python toolkit for reproducing scientific figures and injecting ML classifiers into proteomics data from the VAT T2D study.

## Features

- **Reproducer Tool**: Regenerate publication figures (PCA, volcano, heatmap, QC plots)
- **ML Injector Tool**: Train and evaluate ML classifiers for DM vs NDM prediction
- **Modular Architecture**: Shared core backend supporting future extensions
- **Scientific Visualization**: Publication-grade plots using Matplotlib
- **Optional GUIs**: PySide6/QML frontends for interactive viewing

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Reproducer

```bash
python -m apps.reproducer.cli \
  --input data/raw/41598_2021_96995_MOESM1_ESM.xlsx \
  --out outputs/reproducer
```

Output: `outputs/reproducer/` containing figures and `results.json`

### ML Injector

```bash
python -m apps.injector.cli \
  --input data/raw/41598_2021_96995_MOESM1_ESM.xlsx \
  --out outputs/injector
```

Output: `outputs/injector/` containing ML plots and `metrics.json`, `top_features.json`

## Project Structure

```
t2dvat-suite/
├── t2dvat_core/          # Shared backend modules
│   ├── io.py             # Data loading and I/O
│   ├── preprocessing.py  # Filtering and normalization
│   ├── stats.py          # Differential expression analysis
│   ├── plots.py          # Scientific visualizations
│   ├── ml.py             # Machine learning pipeline
│   └── utils.py          # Utility functions
├── apps/
│   ├── reproducer/       # Figure reproduction tool
│   └── injector/         # ML classification tool
├── data/                 # Input data directory
├── outputs/              # Generated results and figures
└── tests/                # Unit tests
```

## Data Requirements

Input file: `41598_2021_96995_MOESM1_ESM.xlsx` (Scientific Reports 2021)

Required columns:
- `Accession`: Protein ID
- `Gene.name`: Gene symbol
- `Name`: Protein description
- `Abundance.*`: Sample abundance values

Sample grouping is inferred from column names (DM/NDM).

## Development

Run tests:
```bash
pytest tests/
```

## References

- Original paper: *Scientific Reports* 2021, VAT T2D proteomics study
- Data: Supplementary Table S1 (MOESM1_ESM.xlsx)
