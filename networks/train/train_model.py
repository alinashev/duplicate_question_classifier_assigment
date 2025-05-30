import torch
import numpy as np
from torch import nn
from tqdm import tqdm

from metrics.model_metrics import ModelEvaluation
from metrics.display import (
    display_confusion_matrix,
    display_metrics,
    display_roc_auc,
    display_training_curves
)


def choose_device(preferred_device=None) -> torch.device:
    """Chooses an appropriate device for computation (CPU or GPU).

    Args:
        preferred_device (str, optional): Desired device (e.g., "cuda", "cpu").
            If None, CUDA is used if available.

    Returns:
        torch.device: The selected torch device.
    """
    if preferred_device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(preferred_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            print("CUDA not available. Falling back to CPU.")
            device = torch.device("cpu")
    return device


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): Dataloader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        criterion (callable): Loss function.
        device (torch.device): Device to perform computation on.

    Returns:
        tuple:
            - float: Average training loss.
            - list: Ground-truth labels.
            - list: Binary predictions.
            - list: Prediction probabilities.
    """
    model.train()
    total_loss = 0.0
    all_labels, all_probs = [], []

    progress_bar = tqdm(train_loader, desc="Training", leave=True, dynamic_ncols=True,
                        bar_format="{l_bar}{bar} [Elapsed: {elapsed} | Remaining: {remaining}]")

    for batch in progress_bar:
        q1_batch = batch["question1"].to(device)
        q2_batch = batch["question2"].to(device)
        labels = batch["label"].to(device).float()

        optimizer.zero_grad()
        outputs = model(q1_batch, q2_batch)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(outputs.detach().cpu().numpy())

        avg_loss = total_loss / (len(all_probs) or 1)
        progress_bar.set_postfix(loss=f"{avg_loss:.4f}")

    predictions = (np.array(all_probs) > 0.5).astype(int)
    avg_loss = total_loss / len(train_loader)
    return avg_loss, all_labels, predictions, all_probs


def evaluate_model(model, data_loader, criterion, device):
    """Evaluates the model on validation or test data.

    Args:
        model (torch.nn.Module): Trained model.
        data_loader (DataLoader): Dataloader for evaluation.
        criterion (callable): Loss function.
        device (torch.device): Computation device.

    Returns:
        tuple:
            - float: Average validation loss.
            - list: Ground-truth labels.
            - list: Binary predictions.
            - list: Prediction probabilities.
    """
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    with torch.no_grad():
        for batch in data_loader:
            q1_batch = batch["question1"].to(device)
            q2_batch = batch["question2"].to(device)
            labels = batch["label"].to(device)

            outputs = model(q1_batch, q2_batch)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = outputs.cpu().numpy().flatten()
            labels_np = labels.cpu().numpy().flatten()

            all_labels.extend(labels_np)
            all_probs.extend(probs)

    predictions = (np.array(all_probs) > 0.5).astype(int)
    avg_loss = total_loss / len(data_loader)
    return avg_loss, all_labels, predictions, all_probs


def train_model(model, train_loader, val_loader, num_epochs=5, lr=1e-3, device=None, patience=3):
    """Trains the model and evaluates it on validation set with early stopping.

    Args:
        model (torch.nn.Module): Model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        num_epochs (int, optional): Number of training epochs. Defaults to 5.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        device (str or torch.device, optional): Device to use. Defaults to None (auto).
        patience (int, optional): Early stopping patience. Defaults to 3.

    Returns:
        tuple:
            - torch.nn.Module: Trained model with best weights.
            - dict: Training history with losses and validation F1 scores.
    """
    device = choose_device(device)
    print(f"Using device: {device}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_f1 = 0
    best_model_state = None
    early_stopping_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_f1": []
    }

    for epoch in range(num_epochs):
        train_loss, train_labels, train_preds, train_probs = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_labels, val_preds, val_probs = evaluate_model(
            model, val_loader, criterion, device
        )

        eval_metrics = ModelEvaluation(
            y_train=train_labels, train_prediction=train_preds, train_pr_proba=train_probs,
            y_val=val_labels, val_prediction=val_preds, val_pr_proba=val_probs
        )
        val_f1 = next((m.val_value for m in eval_metrics.compute_all_metrics() if m.name == "f1_score"), 0)

        print(f"\n|Epoch {epoch + 1}/{num_epochs}| Train Loss: {train_loss:.4f}; "
              f"Val Loss: {val_loss:.4f}; Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict()
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(f"EarlyStopping patience: {early_stopping_counter}/{patience}")
            if early_stopping_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

    print(f"\nBest Val F1: {best_val_f1:.4f}")
    if best_model_state:
        model.load_state_dict(best_model_state)

    display_metrics(eval_metrics)
    display_confusion_matrix(
        y_train=train_labels, y_val=val_labels,
        train_pred=train_preds, val_pred=val_preds
    )
    display_roc_auc(
        y_train=train_labels, y_val=val_labels,
        train_pr_proba=train_probs, val_pr_proba=val_probs
    )
    display_training_curves(history)

    return model, history
