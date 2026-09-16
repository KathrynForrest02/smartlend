# SmartLend - COMP10086 Artificial Intelligence Applications Project

SmartLend is a fictional UK fintech company building a loan default prediction service.

This repository is the running lab project for **COMP10086 Artificial Intelligence Applications**. You will use it to build a machine learning solution for predicting loan defaults, while also getting experience with the practical side of developing and organising an ML project.

## Project Structure

The repository is organised into the following folders:

```text
smartlend/
│
├── config/                  # Configuration files
├── data/                    # Project data
│   └── raw/                 # Original datasets
├── docs/                    # Project documentation
├── models/                  # Saved trained models
├── notebooks/               # Jupyter notebooks
├── src/                     # Project source code
├── tests/                   # Automated tests
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Folders

### `config/`

Configuration files used by the project.

### `data/`

Data used during the project. The original dataset goes in `data/raw/`. Keep the raw data unchanged.

### `docs/`

Documentation for the project, including any reports, diagrams, or other supporting material.

### `models/`

Saved machine learning models and related files.

### `notebooks/`

Jupyter notebooks for exploring the data, trying out ideas, visualising results, and experimenting with models.

### `src/`

The main Python code for the project. As your work develops, reusable code should be moved out of notebooks and into this folder.

### `tests/`

Tests for your Python code.

A simple way to think about the split between `notebooks/` and `src/` is:

> **Use notebooks to explore. Use `src/` for code you want to keep and reuse.**

## Dataset

The project uses the **Give Me Some Credit** dataset.

Download the dataset using the links provided on the VLE. You should use:

```text
cs-training.csv
```

Place the file here:

```text
data/raw/cs-training.csv
```


> **Please don't edit the original CSV file.** Any cleaning or preprocessing should be done in your code.

## Python Environment

It is recommended that you use a virtual environment for the project.

Create one with:

```bash
python -m venv .venv
```


### Activate Environment (Windows)

```bash
.venv\Scripts\activate
```

### Activate Environment (Linux / macOS)

```bash
source .venv/bin/activate
```

Install the required packages with:

```bash
pip install -r requirements.txt
```

## Getting Started

Once you have the repository set up and the dataset in the correct location, you can start by exploring the data in the `notebooks/` folder.

As the project progresses, you will build up the different parts of the ML workflow and move appropriate code into `src/`.

Keep the repository tidy as you work. In particular:

- Don't modify the raw dataset.
- Don't put all of your code in notebooks.
- Put reusable Python code in `src/`.
- Put tests in `tests/`.
- Save trained models in `models/`.
- Keep project-specific configuration in `config/`.
- Use `docs/` for useful project documentation.

## Project Goal

The aim of the project is to build a **loan default prediction system for SmartLend**.

You will work through the different stages of the machine learning process, from understanding and preparing the data through to training and evaluating models.

The project is also about how you build the system, not just the final model. Your code should be organised, testable, and reproducible, and you should be able to explain the decisions you make along the way.