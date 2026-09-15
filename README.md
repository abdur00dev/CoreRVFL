# CoreRVFL: Correntropy-Based Robust Learning for RVFL Networks

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of **CoreRVFL**, together with its two robust variants **CoreRVFLa** and **CoreRVFLb**, from the paper:

> **Correntropy-Based Robust Learning**  
> A. Rahaman, A. Quadir, and M. Tanveer  
> Department of Mathematics, Indian Institute of Technology Indore, India

CoreRVFL is designed for classification when the data may contain noise, outliers, or unreliable observations. It keeps the lightweight randomized structure of an RVFL network, while replacing ordinary least-squares learning with a robust **maximum total correntropy (MTC)** formulation. Two additional variants combine MTC with M-estimation to reduce sensitivity to the correntropy kernel width and further suppress noisy samples.

---

## Overview

Standard RVFL networks are fast because their hidden-layer parameters are randomly generated and only the output weights are learned. However, the usual ridge-regression objective can be sensitive to outliers and non-Gaussian disturbances.

This repository contains three robust extensions:

| Method | Main idea |
|---|---|
| **CoreRVFL** | Uses the maximum total correntropy criterion to handle disturbances in both the input and output spaces. |
| **CoreRVFLa** | Combines MTC with an **additive M-estimator** weighting mechanism. |
| **CoreRVFLb** | Combines MTC with a **multiplicative M-estimator** weighting mechanism. |

The output weights are obtained through a fixed-point reweighted least-squares procedure, so the models retain the simple and efficient training style of RVFL networks.

---

## Repository structure

```text
CoreRVFL/
├── MTC_RVFL_func.py       # RVFL, CoreRVFL, CoreRVFLa, and CoreRVFLb models
├── MTC_RVFL_main.py       # Cross-validation, grid search, and experiment driver
├── Supplementary.pdf      # Supplementary material for the paper
├── LICENSE                # MIT License
├── README.md
│
├── dataset/               # Create this folder and place datasets here
│   ├── Binary/
│   │   ├── Small/
│   │   └── Large/
│   └── Multi/
│       ├── Small/
│       └── Large/
│
└── results/               # Created automatically when experiments are run
```

The current experiment driver has the two binary-data folders enabled by default. The multiclass folders can be enabled by uncommenting them in `MTC_RVFL_main.py`.

---

## Model names in the code

For clarity, the implementation currently uses the following function names:

| Paper name | Python function |
|---|---|
| CoreRVFL | `MTC_RVFL_Model` |
| CoreRVFLa | `MMTC_RVFLa_Model` |
| CoreRVFLb | `MMTC_RVFLb_Model` |

---

## Requirements

The code was developed for **Python 3.11**.

Install the required packages with:

```bash
pip install numpy scipy scikit-learn joblib
```

Main dependencies:

- NumPy
- SciPy
- scikit-learn
- joblib

No GPU is required.

---

## Dataset format

The experiment script reads MATLAB `.mat` files.

Each dataset is expected to contain a two-dimensional numerical array in which:

- each row corresponds to one sample,
- all columns except the last are input features,
- the last column contains the class label.

For example:

```text
feature_1   feature_2   ...   feature_d   class_label
0.52        1.31        ...   0.18        0
0.47        0.92        ...   0.26        1
...
```

Place the `.mat` files in the appropriate dataset folder, for example:

```text
dataset/Binary/Small/
```

Labels are encoded automatically using `LabelEncoder`, and feature standardization is performed independently inside each cross-validation fold using `StandardScaler`.

---

## Running the experiments

Clone the repository:

```bash
git clone https://github.com/abdur00dev/CoreRVFL.git
cd CoreRVFL
```

Place the datasets inside the `dataset/` directory and run:

```bash
python MTC_RVFL_main.py
```

The script:

1. loads each `.mat` dataset,
2. applies stratified cross-validation,
3. standardizes the features using statistics from the training fold,
4. evaluates the selected model over the hyperparameter grid,
5. uses all available CPU cores through `joblib`,
6. stores the best configuration and its accuracy/runtime statistics.

Results are written to:

```text
results/MTC_RVFL_Results_Joblib.txt
```

---

## Hyperparameter search

The supplied experiment driver searches over:

```text
C              = 10^-5, 10^-4, ..., 10^4, 10^5
Hidden nodes   = 3, 23, 43, ..., 203
Kernel width σ = {0.3, 0.5, 1.0}
M-estimator    = {Huber, Cauchy, Bisquare, Welsch}
Activation     = {Sigmoid, Sine, Tribas, Radbas, Tanh, ReLU}
```

The activation-function indices used in the code are:

| Index | Activation |
|---:|---|
| 1 | Sigmoid |
| 2 | Sine |
| 3 | Triangular basis (`tribas`) |
| 4 | Radial basis (`radbas`) |
| 5 | Hyperbolic tangent (`tanh`) |
| 6 | ReLU |

The regularization parameter used by the solver is

\[
\lambda = \frac{1}{C}.
\]

---

## M-estimators

CoreRVFLa and CoreRVFLb support four commonly used M-estimators:

| Estimator | Tuning constant |
|---|---:|
| Huber | 1.345 |
| Cauchy | 2.385 |
| Bisquare | 4.685 |
| Welsch | 2.985 |

The scale is estimated adaptively from the residual magnitudes during training.

---

## Using a model directly

The model functions can also be called without running the full grid-search script.

```python
from MTC_RVFL_func import MTC_RVFL_Model

option = {
    "C": 1.0,
    "N": 103,
    "sigma": 0.5,
    "activation": 1,
    "max_iter": 30,
    "tol": 1e-3,
}

train_acc, test_acc, train_time, test_time = MTC_RVFL_Model(
    trainX,
    trainY,
    testX,
    testY,
    option,
    No_of_class
)

print(f"Train accuracy: {train_acc:.2f}%")
print(f"Test accuracy : {test_acc:.2f}%")
```

For the M-estimator variants, add:

```python
option["m_estimator"] = "cauchy"
```

and use either:

```python
MMTC_RVFLa_Model(...)
```

or

```python
MMTC_RVFLb_Model(...)
```

---

## Reproducibility notes

The experiment script fixes the NumPy random seed and uses a fixed random state for stratified splitting. Feature normalization is fitted only on the training portion of each fold and then applied to the corresponding test portion.

The current driver uses:

```python
np.random.seed(42)
StratifiedKFold(..., shuffle=True, random_state=42)
```

For very small datasets, the number of folds is automatically reduced when a class contains fewer than five samples.

> **Important:** Before using this repository for exact paper-level reproduction, make sure that the experiment driver matches the final experimental protocol of the paper, including the intended fixed-point iteration limit and the hyperparameter-selection procedure.

---

## Reported results

The paper evaluates the proposed methods on **55 UCI classification datasets**: 30 multiclass and 25 binary datasets.

Reported average classification accuracies are:

| Setting | CoreRVFL | CoreRVFLb | CoreRVFLa |
|---|---:|---:|---:|
| 30 multiclass datasets | 86.07% | 86.10% | **86.34%** |
| 25 binary datasets | 89.42% | 89.28% | **89.67%** |

Please refer to the paper and `Supplementary.pdf` for the complete dataset-wise results, statistical comparisons, and robustness experiments.

---

## Supplementary material

The repository includes:

```text
Supplementary.pdf
```

It contains the unified training algorithm, experimental settings, additional dataset-wise results, statistical tests, and robustness analyses.

---

## Citation

If you use this code or the proposed models in your research, please cite:

> A. Rahaman, A. Quadir, and M. Tanveer, **“Correntropy-Based Robust Learning.”**

The full BibTeX entry can be added here once the final publication metadata is available.

---

## License

This project is released under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions related to the implementation or the paper, please feel free to open a GitHub issue or contact:

**A. Rahaman**  
Department of Mathematics, Indian Institute of Technology Indore, India  
📧 **Email:** [phd2401141001@iiti.ac.in](mailto:phd2401141001@iiti.ac.in) or [abdurrhamanx@gmail.com](mailto:abdurrhamanx@gmail.com)

Repository:  
https://github.com/abdur00dev/CoreRVFL
