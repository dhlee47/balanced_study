# Installation Guide

## Prerequisites

- **Python 3.10 or higher** — download from [python.org](https://www.python.org/downloads/).
  During installation on Windows, check the box that says **"Add Python to PATH"**.
- **Git** (optional) — only needed if you want to clone the repository from the command line.
  You can also just download a ZIP from GitHub (green **Code** button → **Download ZIP**).

---

## Step-by-step Setup

### 1. Get the project folder onto your computer

**Option A — Download ZIP (no Git required):**
1. Go to the repository page on GitHub and click the green **Code** button → **Download ZIP**.
2. Unzip the downloaded file somewhere convenient, e.g. your Desktop or Documents folder.
   You should end up with a folder called `balanced_study-main` (or similar).

**Option B — Clone with Git:**
```
git clone https://github.com/dhlee47/balanced_study.git
```

### 2. Open a terminal inside the project folder

- **Windows:** Open the `balanced_study` folder in File Explorer, then click in the address bar,
  type `cmd`, and press Enter. A Command Prompt window will open already inside that folder.
- **Mac:** Right-click the folder in Finder → **New Terminal at Folder**.
- **Linux:** Right-click the folder → **Open Terminal**.

All the commands below assume your terminal is already inside the `balanced_study` folder.
You can confirm this by typing `dir` (Windows) or `ls` (Mac/Linux) — you should see files like
`requirements.txt` and a folder called `src`.

### 3. Create a virtual environment

A virtual environment keeps this project's dependencies separate from the rest of your computer.
You only need to do this once.

```
python -m venv venv
```

### 4. Activate the virtual environment

You need to do this every time you open a new terminal to use the tool.

```
# Windows Command Prompt
venv\Scripts\activate.bat

# Windows PowerShell
venv\Scripts\Activate.ps1

# Mac / Linux
source venv/bin/activate
```

After activation you will see `(venv)` at the start of your command prompt. That means it worked.

### 5. Install dependencies

```
pip install -r requirements.txt
```

This downloads all the required libraries (~10–15 minutes on a slow connection; faster on subsequent runs).

### 6. Launch the GUI

```
python src\gui.py        # Windows
python src/gui.py        # Mac / Linux
```

A desktop window will open. You are ready to go.

---

## Conda Alternative

If you use Anaconda or Miniconda instead of plain Python:

```
conda env create -f environment.yml
conda activate balanced_study
python src/gui.py
```

---

## Optional: Benchmark and Tests

Run the benchmark across all 10 datasets:
```
python benchmark\run_benchmark.py      # Windows
python benchmark/run_benchmark.py      # Mac / Linux
```
Results appear in `benchmark/results/`.

Run the unit tests:
```
pip install pytest
pytest tests/
```

---

## Troubleshooting

**"Python is not recognised" or "python not found":**
Python was not added to your PATH during installation. Re-run the Python installer and check
the **"Add Python to PATH"** box, or use `py` instead of `python` on Windows.

**PyQt6 import error:**
```
pip install --upgrade PyQt6 PyQt6-Qt6 PyQt6-sip
```

**kaleido not found (plotly static export):**
```
pip install kaleido
```

**statsmodels MANOVA error on very small datasets:**
This is expected — the tool automatically falls back to the permutation test when the dataset
is too small for MANOVA. No action needed.
