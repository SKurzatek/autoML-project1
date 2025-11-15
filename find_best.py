import sklearn
import json
import utils
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import FunctionTransformer

TEST_SIZE = 0.2
RANDOM_STATE = 42

preprocessor = utils.build_two_stage_preprocessor()
DATASETS = ["car", "aps"]
parameters = json.load(open("save.json", "r"))
i = 4
for model in parameters["saves"][i:]:
    for dataset in DATASETS:
        print(f"Testing parameters {i} on {dataset}")
        dataset_info = utils.DATA[dataset]
        file_path = utils.DATA_DIR / dataset_info["file"]
        target = dataset_info["target"]
        df = pd.read_csv(file_path, na_values = dataset_info.get("na_values"))
        X, y = utils.split_xy(df, target)
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = TEST_SIZE, stratify = y, random_state = RANDOM_STATE)

        best_params = { key.replace("model__", "").replace("__", ""): value for key, value in model["best_params"].items()}
        
        to_float = FunctionTransformer(lambda X: X.astype(np.float64))

        best_params['solver'] = "adam"
        mlp_base = MLPClassifier(
            **best_params
        )
        mlp_default = MLPClassifier()

        pipe = Pipeline([
            ("prep", preprocessor),
            ("model",  mlp_base),
        ])
        default_pipe = Pipeline([
            ("prep", preprocessor),
            ("model",  mlp_default),
        ])
        print("Results of custom parameters:")
        print(cross_val_score(pipe, X_test, y_test, cv=5))
        print("Results of default parameters:")
        print(cross_val_score(default_pipe, X_test, y_test, cv = 5))
    i+=1
        



