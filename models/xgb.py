from dataset import Dataset
import xgboost as xgb
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from pprint import pprint
import os
import json


def train_pipeline(
    X_train,
    y_train,
    X_val,
    y_val,
    save_path,
    cls_weight=(1.0, 1.0, 1.0),
    device="gpu",
    undersample=False,
    k=4,
    **kwargs,
):
    T = min(k, 20)  # Ideally we would set T=k but we can only fit max 20 lags in GPU memory.
    dataset_train = Dataset(X=X_train, y=y_train, T=T)
    dataset_val = Dataset(X=X_val, y=y_val, T=T)
    # Seems inefficient but we do this to ensure consistency across different models
    X_train = dataset_train.X.flatten(start_dim=-2, end_dim=-1).squeeze(dim=1).numpy()
    y_train = dataset_train.y.numpy()
    if undersample:
        rus = RandomUnderSampler(sampling_strategy="majority")
        X_train, y_train = rus.fit_resample(X_train, y_train)

    X_val = dataset_val.X.flatten(start_dim=-2, end_dim=-1).squeeze(dim=1).numpy()
    y_val = dataset_val.y.numpy()

    dtrain = xgb.DMatrix(
        X_train,
        label=y_train,
        weight=list(map(lambda x: cls_weight[x], y_train.astype(int))),
    )
    dval = xgb.DMatrix(X_val, label=y_val)
    params = {
        "objective": "multi:softmax",
        "num_class": 3,
        "max_depth": 10,
        "eta": 0.1,
        "subsample": 0.8,         # Subsample ratio of the training instances
        "colsample_bytree": 0.8,  # Subsample ratio of columns when constructing each tree
        "eval_metric": "merror",
        "device": device,
    }
    model = xgb.train(params, dtrain)
    y_val_pred = model.predict(dval)

    best_model_path = f"./model/{save_path}.json"
    if not os.path.exists(best_model_path):
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    model.save_model(best_model_path)

    feat_importance = model.get_score(importance_type="weight")

    d = X_train.shape[1]
    if d == 40 * T:
        feature_prefixes = ["ask_price", "ask_qty", "bid_price", "bid_qty"]
    elif d == 20 * T:
        feature_prefixes = ["ask_orderflow", "bid_orderflow"]
    else:
        raise NotImplementedError(f"X_train shape: {X_train.shape}")

    feature_names = list(
        np.array(
            [
                [
                    [f"{prefix}_{i}_lag_{lag}" for prefix in feature_prefixes]
                    for i in range(1, 11)
                ]
                for lag in range(T - 1, -1, -1)
            ]
        ).flatten()
    )
    feat_importance = {feature_names[int(k[1:])]: v for k, v in feat_importance.items()}
    feat_importance = dict(
        sorted(feat_importance.items(), key=lambda x: x[1], reverse=True)
    )
    pprint(feat_importance)
    feat_importance_save_path = (
        f"./feature_importances/{save_path}_feat_importance.json"
    )
    if not os.path.exists(feat_importance_save_path):
        os.makedirs(os.path.dirname(feat_importance_save_path), exist_ok=True)
    with open(feat_importance_save_path, "w") as f:
        json.dump(feat_importance, f)

    return y_val, y_val_pred


def test_pipeline(
    X_test, y_test, save_path, cls_weight=(1.0, 1.0, 1.0), k=4, **kwargs
):
    T = min(k, 20)
    dataset_test = Dataset(X=X_test, y=y_test, T=T)
    # Seems inefficient but we do this to ensure consistency across different models
    X_test = dataset_test.X.flatten(start_dim=-2, end_dim=-1).squeeze(dim=1).numpy()
    y_test = dataset_test.y.numpy()
    dtest = xgb.DMatrix(
        X_test,
        label=y_test,
        weight=list(map(lambda x: cls_weight[x], y_test.astype(int))),
    )
    best_model_path = f"./model/{save_path.replace('test', 'val')}.json"
    model = xgb.Booster()
    model.load_model(best_model_path)
    y_test_pred = model.predict(dtest)

    return y_test, y_test_pred
