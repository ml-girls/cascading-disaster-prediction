# Environment setup (Miniconda + pip alternatives)

This project supports two common workflows: a Conda-based environment (recommended with Miniconda) and a pip/pyproject workflow.

Conda (recommended)

1. Install Miniconda from https://docs.conda.io/en/latest/miniconda.html
2. Create the environment and install packages:

```powershell
conda env create -f environment.yml
conda activate cascading-disaster
```

3. Register the environment kernel for Jupyter (optional):

```powershell
python -m ipykernel install --user --name=cascading-disaster --display-name "Cascading Disaster (py3.12)"
```

Pip / pyproject (venv)

1. Create a venv and activate (Windows):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install editable package and dependencies from `pyproject.toml`:

```powershell
python -m pip install --upgrade pip
pip install -e .
pip install ipykernel nbformat notebook
python -m ipykernel install --user --name=cascading-disaster-py --display-name "Cascading Disaster (venv)"
```

Notes
- If you prefer `jupyter lab` replace `notebook` with `jupyterlab` in the commands above.
- The `environment.yml` uses `conda-forge`. If you hit package resolution issues, try updating conda: `conda update -n base -c defaults conda`.
