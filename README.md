# COMP9417 Project

This repository contains a set of machine learning experiments for COMP9417. The code compares three models across several datasets:

- `xRFM`
- `XGBoost`
- `MLP` (TensorFlow / Keras)

The shared tuning logic lives in `hp_script.py`, `xgb_tuning.py`, `xrfm_tuning.py`, and `mlp_tuning.py`. Dataset-specific experiment scripts load data, split it into train/test sets, run Bayesian hyperparameter tuning with Optuna, and save the final results under `results/`.

## Repository Layout

- `test_bikesharing.py`: Bike sharing regression experiment
- `test_phishing.py`: Phishing website classification experiment
- `test_physics.py`: Lattice physics regression experiment
- `superconductivity.py`: Superconductivity regression experiment
- `main.ipynb`: House prices notebook experiment
- `house_prices.ipynb`: Earlier house prices notebook work
- `wine copy.ipynb`: Wine dataset notebook experiment
- `datasets/`: Input datasets used by the scripts and notebooks
- `results/`: Saved model outputs and summary files

## Setup

Python 3.11+ is recommended.

### 1. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install ucimlrepo jupyterlab
```

Notes:

- `ucimlrepo` is imported by the lattice physics and notebook code, so install it even though it is not currently listed in `requirements.txt`.
- `jupyterlab` is only needed if you want to run the notebooks.

## How to Run the Code

Run all commands from the repository root

### Bike Sharing experiment

```bash
python test_bikesharing.py
```

Outputs are written to `results/bike_sharing/`.

### Phishing experiment

```bash
python test_phishing.py
```

Outputs are written to `results/phishing/`.

### Lattice Physics experiment

```bash
python test_physics.py
```

Outputs are written to `results/lattice_physics/`.

### Superconductivity experiment

```bash
python superconductivity.py
```

Outputs are written to `results/superconductivity/`.

### House Prices notebook

```bash
jupyter lab
```

Then open 

- `house_prices.ipynb`

using `jupyter lab` or `jupyter notebook`.

## Expected Behaviour

Each script:

1. Loads a dataset from `datasets/`
2. Splits the data into training and test sets
3. Tunes `xRFM`, `XGBoost`, and `MLP` hyperparameters with Optuna
4. Evaluates the best model on the held-out test set
5. Saves results as `.pkl` files in the matching `results/<dataset>/` folder

Most scripts also print final metrics to the terminal.

## Notes

- The bonus part of the project is in bonus.py.
- These experiments are computationally expensive because each model is tuned with multiple Optuna trials and cross-validation folds.
- The scripts create their output folders automatically with `os.makedirs(..., exist_ok=True)`.
- The datasets needed by the current scripts are already included in this repository.
