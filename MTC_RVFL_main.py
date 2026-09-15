import os
import itertools
import warnings

import numpy as np
import scipy.io
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from joblib import Parallel, delayed

from MTC_RVFL_func import (
    RVFL_Model,
    MTC_RVFL_Model,
    MMTC_RVFLa_Model,
    MMTC_RVFLb_Model,
)

warnings.filterwarnings("ignore")
np.random.seed(42)

DIRECTORY = os.path.join(".", "dataset", "Multi", "Small")
RESULT_FILE = os.path.join(".", "results", "MTC_RVFL_Results_Joblib.txt")


def initialize_result_file(filepath):
    headers = [
        "DataSetName", "Model",
        "BestMeanTrainAccuracy", "BestStdTrainAccuracy",
        "BestMeanTestAccuracy", "BestStdTestAccuracy",
        "BestMeanTrainTime", "BestStdTrainTime",
        "BestMeanTestTime", "BestStdTestTime",
        "Best_C", "Best_N", "Best_sigma", "Best_m_estimator",
        "Best_activation",
    ]
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\t".join(headers) + "\n")


def build_grid(model_name, search_space):
    """Build the list of option dictionaries to try for one model."""
    Cs = search_space["C"]
    Ns = search_space["N"]
    sigmas = search_space["sigma"]
    mests = search_space["m_estimator"]
    activations = search_space.get("activation", [1])

    if model_name == "MTC_RVFL":
        return [
            {"C": c, "N": int(n), "sigma": s, "activation": act}
            for c, n, s, act in itertools.product(Cs, Ns, sigmas, activations)
        ]

    return [
        {
            "C": c,
            "N": int(n),
            "sigma": s,
            "m_estimator": m,
            "activation": act,
        }
        for c, n, s, m, act in itertools.product(Cs, Ns, sigmas, mests, activations)
    ]


MODEL_FUNCS = {
    "MTC_RVFL": MTC_RVFL_Model,
    "MMTC_RVFLa": MMTC_RVFLa_Model,
    "MMTC_RVFLb": MMTC_RVFLb_Model,
}


def evaluate_single_option(option, X, y, splits, model_fn, num_classes):

    fold_scores = []

    for tr_idx, te_idx in splits:
        scaler = StandardScaler()
        tr_x = scaler.fit_transform(X[tr_idx])
        te_x = scaler.transform(X[te_idx])
        tr_y, te_y = y[tr_idx], y[te_idx]

        tr_a, te_a, tr_t, te_t = model_fn(
            tr_x, tr_y, te_x, te_y, option, num_classes
        )
        fold_scores.append([tr_a, te_a, tr_t, te_t])

    fold_scores = np.asarray(fold_scores)
    
    return {
        "MeanTrainAccuracy": fold_scores[:, 0].mean(),
        "StdTrainAccuracy": fold_scores[:, 0].std(),
        "MeanTestAccuracy": fold_scores[:, 1].mean(),
        "StdTestAccuracy": fold_scores[:, 1].std(),
        "MeanTrainTime": fold_scores[:, 2].mean(),
        "StdTrainTime": fold_scores[:, 2].std(),
        "MeanTestTime": fold_scores[:, 3].mean(),
        "StdTestTime": fold_scores[:, 3].std(),
        "C": option["C"],
        "N": option["N"],
        "sigma": option.get("sigma", np.nan),
        "m_estimator": option.get("m_estimator", "-"),
        "activation": option.get("activation", 1),
    }

def main():
    initialize_result_file(RESULT_FILE)

    target_folders = [
        os.path.join(".", "dataset", "Binary", "Large"),
        os.path.join(".", "dataset", "Binary", "Small"),
    ]

    search_space = {
        "C": [10.0 **(-5)],
        "N": [203],
        "sigma": [0.3],
        "m_estimator": ["huber", "cauchy", "bisquare", "welsch"],
        "activation": range(1, 2),
    }

    # 2. Loop through each category folder one by one
    for current_dir in target_folders:
        print(f"\n{'='*60}")
        print(f"STARTING FOLDER: {current_dir}")
        print(f"{'='*60}")

        if not os.path.exists(current_dir):
            print(f"Warning: Directory '{current_dir}' not found. Skipping...")
            continue

        files = [f for f in os.listdir(current_dir) if f.endswith(".mat")]
        if not files:
            print(f"No .mat files found in '{current_dir}'. Skipping...")
            continue
            
        print(f"Files to process: {files}")

        for file in files:
            data_path = os.path.join(current_dir, file)
            file_data = scipy.io.loadmat(data_path)

            dataset_key = next((k for k in file_data.keys() if not k.startswith("__")), None)
            if dataset_key is None:
                continue

            all_data = file_data[dataset_key]
            X = np.asarray(all_data[:, :-1], dtype=float)
            y = LabelEncoder().fit_transform(all_data[:, -1])
            num_classes = len(np.unique(y))

            class_counts = np.bincount(y)
            n_splits = min(5, int(class_counts.min()))
            if n_splits < 2:
                print(f"Skipping {file}: at least one class has fewer than 2 samples.")
                continue

            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            splits = list(kf.split(X, y))

            for model_name, model_fn in MODEL_FUNCS.items():
                grid = build_grid(model_name, search_space)
                
                folder_name = os.path.relpath(current_dir, start=os.path.join(".", "dataset"))
                display_name = f"{folder_name}/{file}"
                
                print(f"\n[{display_name}] Parallel grid-searching {model_name} over {len(grid)} configurations...")

                results = Parallel(n_jobs=-1)(
                    delayed(evaluate_single_option)(option, X, y, splits, model_fn, num_classes)
                    for option in grid
                )

                best = max(results, key=lambda r: r["MeanTestAccuracy"])

                with open(RESULT_FILE, "a", encoding="utf-8") as fout:
                    row = [
                        display_name, # Saves as "Binary\Small\filename.mat"
                        model_name,
                        f"{best['MeanTrainAccuracy']:.4f}",
                        f"{best['StdTrainAccuracy']:.4f}",
                        f"{best['MeanTestAccuracy']:.4f}",
                        f"{best['StdTestAccuracy']:.4f}",
                        f"{best['MeanTrainTime']:.4f}",
                        f"{best['StdTrainTime']:.4f}",
                        f"{best['MeanTestTime']:.4f}",
                        f"{best['StdTestTime']:.4f}",
                        f"{best['C']:.6f}",
                        str(best["N"]),
                        f"{best['sigma']:.2f}" if not np.isnan(best["sigma"]) else "-",
                        str(best["m_estimator"]),
                        str(best["activation"]),
                    ]
                    fout.write("\t".join(row) + "\n")

                print(
                    f"  -> best test acc {best['MeanTestAccuracy']:.4f}% "
                    f"(C={best['C']:.0e}, N={best['N']}, sigma={best['sigma']}, "
                    f"m_est={best['m_estimator']}, activation={best['activation']})"
                )

    print("\nAll folders processed successfully!")

if __name__ == "__main__":
    main()


