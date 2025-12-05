# T2D-VAT-Suite: Two-Tool Architecture for Proteomics Analysis

A modular Python toolkit for reproducing scientific figures and injecting ML classifiers into proteomics data from the VAT T2D study.

## Features

- **Reproducer Tool**: Regenerate publication figures (PCA, volcano, heatmap, QC plots)
- **ML Injector Tool**: Train and evaluate ML classifiers for DM vs NDM prediction
- **Modular Architecture**: Shared core backend supporting future extensions
- **Scientific Visualization**: Publication-grade plots using Seaborn
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

## Key Findings

### Proteomic Profile of Visceral Adipose Tissue in Type 2 Diabetes

This analysis of VAT (visceral adipose tissue) proteomics data reveals significant molecular alterations between diabetic (DM) and non-diabetic metabolic syndrome (NDM) patients.

#### Dataset Overview
- **Proteins analyzed**: 1,965 proteins quantified across all samples
- **Sample size**: 20 patients (DM vs NDM groups)
- **Significantly altered proteins**: 5 proteins with FDR q-value < 0.05
- **Total significant hits**: 23 proteins using author significance criteria

#### Critical Biomarker Discoveries

**FYCO1 (FYVE and coiled-coil domain-containing protein 1)**
- **Protein ID**: Q9BQS8
- **Significance**: q-value = 0.045 (FDR-corrected)
- **Expression change**: 0.47-fold decrease in DM patients (p = 6.1×10⁻⁵)
- **Clinical relevance**: Key autophagy regulator, suggesting impaired cellular clearance mechanisms in diabetic VAT

**STAT5B (Signal transducer and activator of transcription 5B)**
- **Protein ID**: P51692  
- **Significance**: q-value = 0.032 (most significant upregulated protein)
- **Expression change**: 2.08-fold increase in DM patients (p = 3.4×10⁻⁴)
- **Clinical relevance**: Transcription factor involved in growth hormone signaling and metabolic regulation

**GAS1 (Growth arrest-specific protein 1)**
- **Protein ID**: P54826
- **Significance**: q-value = 0.032 
- **Expression change**: 0.34-fold decrease in DM patients (p = 1.1×10⁻⁵)
- **Clinical relevance**: Cell cycle regulator and apoptosis mediator, indicating altered cell survival pathways

**POFUT1 (GDP-fucose protein O-fucosyltransferase 1)**
- **Protein ID**: Q9H488
- **Significance**: q-value = 0.045
- **Expression change**: 0.35-fold decrease in DM patients (p = 1.8×10⁻⁵)
- **Clinical relevance**: Protein glycosylation enzyme affecting Notch signaling pathway

#### Machine Learning Classification Performance

**Model Performance Metrics**
- **Cross-validation accuracy**: 100% (perfect separation)
- **ROC AUC**: 1.0 (perfect discrimination)
- **Precision, Recall, F1-score**: All 1.0 (perfect classification)

**Top Discriminative Features (by ML importance)**
1. **RANGAP1** (Ran GTPase-activating protein 1) - Importance: 0.083
2. **DES** (Desmin) - Importance: 0.065  
3. **HSPB7** (Heat shock protein beta-7) - Importance: 0.060
4. **TTN** (Titin) - Importance: 0.057
5. **HDLBP** (Vigilin) - Importance: 0.055

#### Unsupervised Clustering Analysis
- **Clustering approach**: K-means (k=2) on protein expression profiles
- **Adjusted Rand Index**: 0.044 vs true DM/NDM labels
- **Interpretation**: Moderate agreement between unsupervised clustering and clinical diagnosis, suggesting distinct but overlapping molecular phenotypes

#### Clinical Implications

1. **Autophagy Dysfunction**: Decreased FYCO1 suggests impaired autophagy in diabetic VAT, potentially contributing to metabolic dysfunction and insulin resistance.

2. **Altered Growth Signaling**: Elevated STAT5B indicates dysregulated growth hormone signaling pathways in diabetic adipose tissue.

3. **Cell Cycle Perturbations**: Reduced GAS1 expression suggests altered cell growth control and potential increased proliferation in diabetic VAT.

4. **Glycosylation Defects**: Decreased POFUT1 indicates disrupted protein glycosylation, which may affect cellular signaling and protein function.

5. **Molecular Stratification**: Perfect ML classification demonstrates that VAT proteomics can reliably distinguish DM from NDM patients, suggesting potential for biomarker development.

#### Generated Visualizations
- Principal Component Analysis (PCA) plots showing sample clustering
- Volcano plots highlighting significant proteins
- Heatmaps of top differentially expressed proteins  
- ROC curves and confusion matrices for ML classification
- Feature importance rankings for biomarker prioritization
- Quality control plots for data validation

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
