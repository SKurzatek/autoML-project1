import utils
from scipy.stats import randint, uniform, loguniform
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import time
from skopt.space import Real, Integer, Categorical

DATASETS = ["jannis"] #["car", "aps", "covertype", "jannis"]
MODEL = "mlp"
N_ITER = 50
CV_SPLITS = 4
RANDOM_STATE = 42
TEST_SIZE = 0.2
SEARCH_METHOD = "random"  # "bayes" or "random"

grid = {
    "mlp": {
        "model__hidden_layer_sizes": [(16,), (32, ), (64, ), (128,), (256,)],
        "model__activation": ["relu", "tanh"],
        "model__solver": ["adam"],
        "model__alpha": loguniform(1e-7, 1e-1),
        "model__learning_rate_init": loguniform(1e-4, 0.5),
        "model__n_iter_no_change": [10, 20],
        "model__tol": [1e-3, 3e-3, 1e-2],
        "model__batch_size": [128, 256, 512]
    }
}

# ... importy
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

# --- DODAJ TĘ KLASĘ ---
class MLPWrapper(MLPClassifier):
    """
    Wrapper dla MLPClassifier, który konwertuje parametr int hidden_layer_sizes
    na krotkę (tuple), aby umożliwić współpracę z BayesSearchCV.
    """
    def set_params(self, **params):
        if "hidden_layer_sizes" in params:
            if isinstance(params["hidden_layer_sizes"], int):
                # Konwertuj int na krotkę z jednym elementem
                params["hidden_layer_sizes"] = (params["hidden_layer_sizes"],)
        return super().set_params(**params)
# ---------------------




print("Checking dataset files:")
for k in DATASETS:
    if utils._exists(k):
        print(f" - {k:10s} -> {utils.DATA[k]['file']}  ✓")
    else:
        print(f" - {k:10s} -> {utils.DATA[k]['file']}  ✗")

print("Building model pipeline...")

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

for dataset in DATASETS:
    dataset_info = utils.DATA[dataset]
    file_path = utils.DATA_DIR / dataset_info["file"]
    df = pd.read_csv(file_path, na_values=dataset_info.get("na_values"))

    X, y = utils.split_xy(df, dataset_info["target"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    le = utils.TargetLabelEncoder().fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)

    preprocessor = utils.build_two_stage_preprocessor()

    mlp_base = MLPWrapper(
        solver="adam",
    )

    models = {
        "mlp": mlp_base,
    }

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model",  models[MODEL]),
    ])
    best_params_r, best_score_r, results_r, rs_obj = None, None, None, None
    if SEARCH_METHOD == "random":
        print(f"Running RandomizedSearchCV for {dataset} with {MODEL}...")
        best_params_r, best_score_r, results_r, rs_obj = utils.run_random_search(
            param_distributions = grid[MODEL],
            pipe = pipe,
            X = X_train,
            y = y_train_enc,
            n_iter = N_ITER,
            cv_splits = CV_SPLITS,
            random_state = RANDOM_STATE,
            n_jobs = -1,
            verbose = 2,
        )
    elif SEARCH_METHOD == "bayes":
        print(f"Running BayesSearchCV for {dataset} with {MODEL}...")
        best_params_r, best_score_r, results_r, rs_obj = utils.run_bayes_search(
            search_spaces = grid[MODEL],
            pipe = pipe,
            X = X_train,
            y = y_train_enc,
            n_iter = N_ITER,
            cv_splits = CV_SPLITS,
            random_state = RANDOM_STATE,
            n_jobs = -1,
            verbose = 2,
        )
    else:
        print(f"Unknown SEARCH_METHOD: {SEARCH_METHOD}")

    save_dict = json.load(open("save.json", "r"))
    save_dict["saves"].append({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "dataset": dataset,
        "model": MODEL,
        "search": SEARCH_METHOD,
        "best_params": best_params_r,
        "best_score": best_score_r
    })
    json.dump(save_dict, open("save.json", "w"), indent=4)