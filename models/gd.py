import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter()


def run_gradient_descent(
    model,
    criterion,
    optimizer,
    train_loader,
    val_loader,
    epochs,
    best_model_path,
):
    """Train a PyTorch model using the given criterion and optimizer with gradient descent."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_losses = np.zeros(epochs)
    val_losses = np.zeros(epochs)
    best_val_loss = np.inf
    best_val_epoch = 0

    for it in tqdm(range(epochs)):
        model.train()
        t0 = datetime.now()
        train_loss = []
        progress_bar = tqdm(train_loader, desc=f"Epoch: {it+1} / {epochs}")
        for j, (X_batch, y_true_batch) in enumerate(progress_bar):
            X_batch, y_true_batch = X_batch.to(
                device, dtype=torch.float
            ), y_true_batch.to(device, dtype=torch.int64)
            optimizer.zero_grad()
            y_pred_batch = model(X_batch)
            loss = criterion(y_pred_batch, y_true_batch)
            loss.backward()
            # Update progress_bar with batch loss
            if j % 50 == 0:
                progress_bar.set_postfix(loss=loss.item())
            optimizer.step()
            train_loss.append(loss.item())

        train_loss = np.mean(train_loss)
        # writer.add_scalar("Loss/train", train_loss, it)

        model.eval()
        val_loss = []
        for X_batch, y_true_batch in val_loader:
            X_batch, y_true_batch = X_batch.to(
                device, dtype=torch.float
            ), y_true_batch.to(device, dtype=torch.int64)
            outputs = model(X_batch)
            loss = criterion(outputs, y_true_batch)
            val_loss.append(loss.item())

        val_loss = np.mean(val_loss)
        # writer.add_scalar("Loss/val", val_loss, it)
        train_losses[it] = train_loss
        val_losses[it] = val_loss
        if val_loss < best_val_loss:
            torch.save(model, best_model_path)
            best_val_loss = val_loss
            best_val_epoch = it
            print("model saved")

        dt = datetime.now() - t0
        print(" ")
        print(" ")
        print(
            f"Epoch {it+1}/{epochs}, Train Loss: {train_loss:.4f}, \
          Validation Loss: {val_loss:.4f}, Duration: {dt}, Best Val Epoch: {best_val_epoch + 1}"
        )
        print(" ")

    # writer.flush()

    return train_losses, val_losses
