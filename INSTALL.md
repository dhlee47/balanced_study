# Installation Guide

## Prerequisites
- Python 3.10 or higher
- Git (optional, for cloning)

## Step-by-step Setup

### 1. Navigate to the project folder
```
cd /path/to/balanced_study      # Linux / Mac
cd C:\path\to\balanced_study    # Windows
```

### 2. Create the virtual environment
```
python -m venv venv
```

### 3. Activate the virtual environment
```
# Windows CMD
venv\Scripts\activate.bat

# Windows PowerShell
venv\Scripts\Activate.ps1

# Mac / Linux
source venv/bin/activate
```

### 4. Install dependencies
```
pip install -r requirements.txt
```

### 5. Run the GUI
```
python src\gui.py
```

### 6. Run the benchmark (optional)
```
python benchmark\run_benchmark.py
```
Results appear in `benchmark\results\`.

### 7. Run unit tests (optional)
```
pip install pytest
pytest tests\
```

## Conda Alternative
```
conda env create -f environment.yml
conda activate balanced_study
python src\gui.py
```

## Troubleshooting

**PyQt6 import error on first run:**
```
pip install --upgrade PyQt6 PyQt6-Qt6 PyQt6-sip
```

**kaleido not found (plotly static export):**
```
pip install kaleido
```

**statsmodels MANOVA error on very small datasets:**
This is expected — the tool automatically falls back to the permutation test when n is too small for MANOVA. No action needed.
