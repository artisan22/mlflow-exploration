# MLflow Exploration

A hands-on introduction to MLflow experiment tracking. Train a simple ML model multiple times with different settings and use MLflow to record, compare, and load your results.

---

## What You'll Learn

- What MLflow is and why it exists
- How to track experiment params and metrics across multiple runs
- How to compare runs in the MLflow UI
- How to save and load a trained model

---

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed
- Python 3.10+
- Git

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/mlflow-exploration.git
cd mlflow-exploration
uv sync
```

---

## How to Run

### 1. Train the model

```bash
uv run python train.py
```

This trains a RandomForest classifier on the Iris dataset and logs the run to MLflow.

To compare different settings, open `train.py` and change these values at the top, then run again:

```python
N_ESTIMATORS = 10   # try 50, 100
MAX_DEPTH = 3       # try 5, None
```

Run it 3 times with different values to get multiple runs to compare.

### 2. View results in the MLflow UI

```bash
uv run mlflow ui
```

Open your browser at `http://127.0.0.1:5000`

- Click **iris-classifier** in the left sidebar
- Click **Training runs**
- Compare params and metrics across all your runs

### 3. Load the model and make a prediction

```bash
uv run python predict.py
```

This loads the most recent trained model from MLflow and predicts the species of a new flower without retraining.

---

## Project Structure

```
mlflow-exploration/
├── train.py          # trains the model and logs to MLflow
├── predict.py        # loads saved model and makes a prediction
├── pyproject.toml    # dependencies managed by uv
└── README.md
```

---

## Key Concepts

| Term | What it means |
|---|---|
| Experiment | A named group of related runs (e.g. "iris-classifier") |
| Run | One execution of train.py with specific settings |
| Params | The settings you chose before training (inputs) |
| Metrics | The results you measured after training (outputs) |
| Artifacts | Files saved from a run (e.g. the trained model file) |

---

## Why MLflow?

Without experiment tracking you quickly lose track of which settings produced the best model. MLflow records everything automatically so you can always answer:

- Which run had the best accuracy?
- What settings did I use for that run?
- Can I reproduce it?

---


