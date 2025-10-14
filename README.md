# POEMS
POEMS: Product of Experts for Interpretable Multi-omic Integration using Sparse Decoding


## Data Organization

The dataset used in this project is located in the `data/` directory and includes two cancer types from **The Cancer Genome Atlas (TCGA)**:  **BRCA** (Breast Invasive Carcinoma) and **KIRC** (Kidney Renal Clear Cell Carcinoma).  Each subfolder contains the multi-omics data used for model training and evaluation.

data/
├── brca/
│   ├── 1_all.csv         → mRNA expression values
│   ├── 1_featname.csv    → mRNA feature names
│   ├── 2_all.csv         → DNA methylation values
│   ├── 2_featname.csv    → DNA methylation feature names
│   ├── 3_all.csv         → miRNA expression values
│   ├── 3_featname.csv    → miRNA feature names
│   └── labels_all.csv    → sample subtype labels
└── kirc/
    ├── gene1.csv.gz         → mRNA expression values (compressed)
    ├── methyl.csv.gz        → DNA methylation values (compressed)
    ├── miRNA1.csv.gz        → miRNA expression values (compressed)
    └── label.csv.gz         → sample subtype labels (compressed)


## Setup

### 1. Clone the repository
```bash
git clone https://github.com/anonymous-fish14/POEMS.git
cd POEMS
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install required dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```