"""
setup.py — Package setup for balanced_study.

Install in development mode: pip install -e .
"""
from setuptools import setup, find_packages

setup(
    name="balanced_study",
    version="1.0.0",
    description="Preclinical in vivo study group balancing toolkit",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.17.0",
        "statsmodels>=0.14.0",
        "pingouin>=0.5.3",
        "scikit-posthocs>=0.8.0",
        "PyQt6>=6.6.0",
        "openpyxl>=3.1.0",
        "tqdm>=4.66.0",
        "kaleido>=0.2.1",
        "reportlab>=4.0.0",
    ],
    entry_points={
        "console_scripts": [
            "balanced-study=gui:main",
        ],
    },
)
