# T2D-VAT-Suite: Two-Tool Architecture & Full Build Plan

This document is meant to be fed directly into an LLM coding agent. It contains:

- High-level reasoning
- Data contracts
- Full module-by-module build sequence
- Folder structure
- CLI spec
- Plotting + ML requirements
- GUI placement

All content is structured to allow an LLM to begin coding without asking additional questions.

---

# 1. Project Overview

We are building **one repository** with **two mini-tools** that share a common backend.

## Tool 1 — Reproducer

Input: Excel protein table from *Scientific Reports* 2021 T2D proteomics paper. Goal: Reproduce core scientific figures using Seaborn. Outputs:

- QC barplots
- PCA
- Volcano
- Heatmap of differential proteins
- Optional enrichment/dotplot
- results.json summarizing DE stats

## Tool 2 — ML Injector

Input: Same protein table. Goal: Inject an ML classifier layer (logistic regression or random forest) to classify DM vs NDM. Outputs:

- ROC curve
- Confusion matrix
- Feature importance plot
- top\_features.json + metrics.json

Common core logic lives in **t2dvat\_core/**.

Both tools get optional **PySide6/QML GUIs** that simply display generated images. They do NOT render plots natively. All figures are exported as PNG/SVG via Seaborn (Matplotlib backend).

---

# 2. Repo Structure

```
t2dvat-t2d-lab/
├─ README.md
├─ pyproject.toml / requirements.txt
├─ data/
│  └─ raw/41598_2021_96995_MOESM1_ESM.xlsx
├─ t2dvat_core/
│  ├─ __init__.py
│  ├─ io.py
│  ├─ preprocessing.py
│  ├─ stats.py
│  ├─ plots.py
│  ├─ ml.py
│  └─ utils.py
├─ apps/
│  ├─ reproducer/
│  │  └─ cli.py
│  └─ injector/
│     └─ cli.py
├─ outputs/
│  ├─ reproducer/
│  └─ injector/
└─ tests/
```

---

# 3. Data Contract (Very Important)

The input file is: `41598_2021_96995_MOESM1_ESM.xlsx` (VAT proteome, DM vs NDM).

## 3.1 Columns needed

- **Accession** — protein ID
- **Gene.name** — gene symbol
- **Name** — protein description
- *Abundance.** columns*\* — one per sample. Example: `Abundance.DM1`, `Abundance.NDM4`.

## 3.2 Sample metadata inference

Group assignment is inferred from column naming:

- If column contains "DM": sample\_group = DM
- If column contains "NDM": sample\_group = NDM

## 3.3 Output of `load_protein_table()`

Return a `ProteinTable` dataclass with:

```
ProteinTable:
  proteins: DataFrame
      index: protein_id
      columns: [gene_name, protein_name]

  X: DataFrame
      index: protein_id
      columns: sample_ids
      values: numeric abundances

  sample_meta: DataFrame
      columns: [sample_id, group]
      group ∈ {"DM", "NDM"}
```

---

# 4. Coding Order (Critical for Agents)

This is the exact order modules should be implemented.

## Phase 1 — `io.py`: Load the dataset

Implement:

- `load_protein_table(path: str) -> ProteinTable`
- `infer_group(col_name: str) -> str`

Must:

- Read Excel file
- Extract protein metadata
- Extract numeric abundance matrix
- Build sample metadata table

Stop only when:

```python
pt = load_protein_table("data/raw/41598_2021_96995_MOESM1_ESM.xlsx")
print(pt.X.shape)
print(pt.sample_meta.groupby("group").size())
```

passes.

---

## Phase 2 — `preprocessing.py`: Filtering + Transforming

Implement:

- `log_transform(X)` → log2(x + 1)
- `filter_low_abundance(X, min_valid_samples=4)`
- `make_analysis_ready(pt: ProteinTable)` → returns updated pt

Stop only when printed shapes look correct.

---

## Phase 3 — `stats.py`: Differential Expression

Implement:

- `compute_differential_stats(pt: ProteinTable) -> pd.DataFrame`

Must compute:

- mean\_DM
- mean\_NDM
- log2\_fc
- p-value (t-test)
- q-value (FDR correction)

Output DataFrame index = protein\_id.

Stop only when you can print:

```
(sum(diff.q_value < 0.05))
```

---

## Phase 4 — `plots.py`: Seaborn Scientific Figures

Implement:

### 4.1 QC

- `plot_total_intensity_per_sample(pt, out_path)`
- `plot_protein_counts_per_sample(pt, out_path)`

### 4.2 PCA

- `plot_pca(pt, out_path)`

### 4.3 Volcano

- `plot_volcano(diff_df, out_path)`

### 4.4 Heatmap

- `plot_topN_heatmap(pt, diff_df, N, out_path)`

### 4.5 Optional Pathway Plot (stub)

- `plot_pathway_enrichment(diff_df, out_path)`

All functions must save PNG.

Stop only when running these produces figures in `outputs/reproducer/figures/`.

---

## Phase 5 — Reproducer CLI (apps/reproducer/cli.py)

Implement command:

```bash
python -m apps.reproducer.cli \
  --input data/raw/41598_2021_96995_MOESM1_ESM.xlsx \
  --out outputs/reproducer
```

CLI pipeline:

1. Load table
2. Preprocess
3. Compute stats
4. Generate plots
5. Write results.json

Stop only when `outputs/reproducer/` is fully populated.

---

## Phase 6 — ML (ml.py): Classifier

Implement:

- `make_ml_matrix(pt: ProteinTable) -> (X, y, feature_meta)`
- `train_classifier(X, y)` using LogisticRegression or RandomForest
- Produce:
  - cross-validated metrics
  - fitted model
  - coefficient importances

Add plots:

- `plot_roc_curve(...)`
- `plot_confusion_matrix(...)`
- `plot_feature_importance_topN(...)`

Stop only when ML plots appear under `outputs/injector/figures/`.

---

## Phase 7 — Injector CLI (apps/injector/cli.py)

Implement command:

```bash
python -m apps.injector.cli \
  --input data/raw/41598_2021_96995_MOESM1_ESM.xlsx \
  --out outputs/injector
```

CLI pipeline:

1. Load table
2. Preprocess
3. Build ML matrix
4. Train classifier
5. Save metrics + ML plots

Stop when both results.json + figures exist.

---

## Phase 8 — GUI (QML)

### GUI Principle

QML **does not** generate scientific plots. It simply **embeds** Seaborn-generated PNG/SVG files.

### Reproducer GUI

- Loads results.json from `outputs/reproducer/`
- Displays PCA, volcano, heatmap, QC plots via:

```
Image {
  source: "file:///absolute/path/to/figure.png"
  fillMode: Image.PreserveAspectFit
}
```

### Injector GUI

- Loads ML metrics
- Displays ROC, confusion matrix, feature importance
- Includes table of top predictive proteins

---

# 5. Tool Commands Summary

### Reproducer

```
python -m apps.reproducer.cli \
  --input path/to/excel \
  --out outputs/reproducer
```

### Injector

```
python -m apps.injector.cli \
  --input path/to/excel \
  --out outputs/injector
```

---

# 6. Seaborn Styling Requirements

LLM should enforce:

- high DPI (300+)
- transparent=True for GUI embedding
- consistent font (Roboto/Helvetica)
- color palettes suitable for scientific graphics
- axis spine cleanup
- publication-grade volcano & PCA layouts

---

# 7. End State

When both tools are complete, the repository provides:

- **Automatic reproduction** of the paper’s key figures
- **Automatic ML injection** for classification
- **Two QML frontends** showing rendered plots
- A clean, modular backend that supports future datasets

This document contains the full blueprint for an LLM coding agent to implement the system end-to-end.
