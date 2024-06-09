from dataset import Dataset
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler
import joblib
import os


def train_pipeline(
    X_train,
    y_train,
    X_val,
    y_val,
    save_path,
    cls_weight=(1.0, 1.0, 1.0),
    undersample=False,
    **kwargs,
):
    T = 10
    dataset_train = Dataset(X=X_train, y=y_train, T=T)
    dataset_val = Dataset(X=X_val, y=y_val, T=T)
    # Seems inefficient but we do this to ensure consistency across different models
    X_train = dataset_train.X.flatten(start_dim=-2, end_dim=-1).squeeze(dim=1)
    y_train = dataset_train.y

    if undersample:
        rus = RandomUnderSampler(sampling_strategy="majority")
        X_train, y_train = rus.fit_resample(X_train, y_train)

    X_val = dataset_val.X.flatten(start_dim=-2, end_dim=-1).squeeze(dim=1)
    y_val = dataset_val.y
    # Instantiate model
    model = LogisticRegression(
        class_weight={c: cls_weight[c] for c in range(len(cls_weight))},
        max_iter=100,
    )
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)

    model_path = f"./model/{save_path}.pkl"
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

    joblib.dump(model, model_path)

    return y_val, y_val_pred


def test_pipeline(X_test, y_test, save_path, cls_weight=(1.0, 1.0, 1.0), T=10, **kwargs):
    dataset_test = Dataset(X=X_test, y=y_test, T=T)
    # Seems inefficient but we do this to ensure consistency across different models
    X_test = dataset_test.X.flatten(start_dim=-2, end_dim=-1).squeeze(dim=1).numpy()
    y_test = dataset_test.y.numpy()
    model_path = f"./model/{save_path.replace('test', 'val')}.pkl"
    model = joblib.load(model_path)
    y_test_pred = model.predict(X_test)
    return y_test, y_test_pred
