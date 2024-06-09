"""
Original src: https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books
"""
import torch
import torch.nn as nn
import numpy as np
from dataset import Dataset
from gd import run_gradient_descent
import gc
import os


class DeepOF(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=(1, 2), stride=(1, 2)
            ),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 10)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )

        # Inception modules
        self.inp1 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(1, 1), padding="same"
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=(3, 1), padding="same"
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(1, 1), padding="same"
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=(5, 1), padding="same"
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(1, 1), padding="same"
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=192, hidden_size=64, num_layers=1, batch_first=True
        )
        self.fc1 = nn.Linear(64, 3)

    def forward(self, x):
        # h0: (number of hidden layers, batch size, hidden size)
        h0 = torch.zeros(1, x.size(0), 64).to(self.device)
        c0 = torch.zeros(1, x.size(0), 64).to(self.device)
        x = self.conv2(x)
        x = self.conv3(x)
        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)
        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))
        x, _ = self.lstm(x, (h0, c0))
        x = x[:, -1, :]
        x = self.fc1(x)
        forecast_y = torch.softmax(x, dim=1)
        # forecast_y = torch.log_softmax(x, dim=1)

        return forecast_y

    def predict(self, dataloader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        y_pred = []
        y_true = []
        self.eval()
        for X_batch, y_true_batch in dataloader:
            X_batch = X_batch.to(device, dtype=torch.float)
            y_pred_batch = self(X_batch)
            # torch.max returns both max and argmax
            _, y_pred_batch = torch.max(y_pred_batch, 1)
            y_pred.append(y_pred_batch)
            y_true.append(y_true_batch)

        y_pred = torch.concatenate(y_pred).cpu().numpy()
        y_true = torch.concatenate(y_true).cpu().numpy()

        return y_true, y_pred


def train_pipeline(
    X_train,
    y_train,
    X_val,
    y_val,
    save_path,
    cls_weight=(1.0, 1.0, 1.0),
    batch_size=64,
    learning_rate=1e-4,
    epochs=5,
    warm_start=False,
    **kwargs,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_train = Dataset(X=X_train, y=y_train, T=100)
    dataset_val = Dataset(X=X_val, y=y_val, T=100)
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset_train, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=dataset_val, batch_size=batch_size, shuffle=False
    )
    best_model_path = f"./model/{save_path}.pt"
    if not os.path.exists(best_model_path):
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    if warm_start:
        model = torch.load(best_model_path)
    else:
        model = DeepOF()

    model.to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(cls_weight).to(device))
    # criterion = torch.nn.NLLLoss(weight=torch.tensor(cls_weight).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_losses, val_losses = run_gradient_descent(
        model,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        epochs,
        best_model_path,
    )

    best_model = torch.load(best_model_path)
    y_val, y_val_pred = best_model.predict(val_loader)

    losses_path = f"./losses/{save_path}.npy"
    if not os.path.exists(losses_path):
        os.makedirs(os.path.dirname(losses_path), exist_ok=True)

    losses = np.vstack((train_losses, val_losses))
    np.save(losses_path, losses)

    del model
    del dataset_train
    del dataset_val
    del train_loader
    del val_loader
    del X_train
    del y_train
    del X_val
    torch.cuda.empty_cache()
    gc.collect()

    return y_val, y_val_pred


def test_pipeline(
    X_test,
    y_test,
    save_path,
    batch_size=64,
    **kwargs,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_test = Dataset(X=X_test, y=y_test, T=100)
    test_loader = torch.utils.data.DataLoader(
        dataset=dataset_test, batch_size=batch_size, shuffle=True
    )
    best_model_path = f"./model/{save_path.replace('test', 'val')}.pt"
    model = torch.load(best_model_path)
    model.eval()
    model.to(device)
    y_test, y_test_pred = model.predict(test_loader)
    del model
    del dataset_test
    del test_loader
    del X_test
    torch.cuda.empty_cache()
    gc.collect()

    return y_test, y_test_pred
