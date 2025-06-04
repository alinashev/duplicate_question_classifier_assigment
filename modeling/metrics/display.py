from matplotlib import pyplot as plt

from rich.console import Console
from rich.table import Table

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc


def display_metrics(metrix_obj):
    """
        Displays a table of model explaining metrics using the `rich` library.

        Args:
            metrix_obj: An object that implements the `compute_all_metrics()` method,
                        which returns a list of metrics with `name`, `train_value`, and `val_value` attributes.

        Returns:
            None
    """

    metrics = metrix_obj.compute_all_metrics()

    table = Table(title="Model Evaluation Metrics")
    table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
    table.add_column("Train Value", justify="right", style="green")
    table.add_column("Validation Value", justify="right", style="magenta")

    for metric in metrics:
        table.add_row(
            metric.name,
            f"{metric.train_value:.4f}" if metric.train_value is not None else "-",
            f"{metric.val_value:.4f}" if metric.val_value is not None else "-"
        )

    console = Console()
    console.print(table)


def display_cross_validation_result(cv_results):
    """
        Displays cross-validation metrics including mean scores and standard deviations in a formatted table.

        Args:
            cv_results (dict): A dictionary where keys are metric names and values are dicts with keys
                               'mean' and 'std' representing the average score and its standard deviation.

        Returns:
            None
    """

    cv_table = Table(title="Cross-validation Metrics")
    cv_table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
    cv_table.add_column("Mean Score", justify="right", style="green")
    cv_table.add_column("Standard Deviation", justify="right", style="yellow")

    for metric, result in cv_results.items():
        cv_table.add_row(
            metric,
            f"{result['mean']:.4f}",
            f"{result['std']:.4f}"
        )
    console = Console()
    console.print("\nCross-validation Results:")
    console.print(cv_table)


def _plot_confusion_matrix(y_true, y_pred, title, ax):
    """
    Helper function to plot a single confusion matrix on the given axis.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        title (str): Title for the subplot (e.g., 'Train' or 'Validation').
        ax (matplotlib.axes.Axes): The axis on which to plot the confusion matrix.

    Returns:
        None
    """

    cm = confusion_matrix(y_true, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f"{title} Confusion Matrix")


def display_confusion_matrix(y_train, y_val, train_pred, val_pred):
    """
    Displays the confusion matrix for both training and validation sets.

    Args:
        y_train (array-like): Ground truth labels for the training set.
        y_val (array-like): Ground truth labels for the validation set.
        train_pred (array-like): Predicted labels for the training set.
        val_pred (array-like): Predicted labels for the validation set.

    Returns:
        None
    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    _plot_confusion_matrix(y_train, train_pred, "Train", axes[0])
    _plot_confusion_matrix(y_val, val_pred, "Validation", axes[1])
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_roc_curve(fpr, tpr, roc_auc, name):
    """Plots the Receiver Operating Characteristic (ROC) curve.

    Args:
        fpr (array-like): False Positive Rates.
        tpr (array-like): True Positive Rates.
        roc_auc (float): Area Under the Curve (AUC) value.
        name (str): Name of the dataset or model for the plot title.
    """
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.4f})', color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def plot_roc_auc(y_true, y_proba, pos_label, dataset_name, ax=None):
    """Computes and plots the ROC curve with AUC for classification predictions.

    Args:
        y_true (array-like): True binary labels.
        y_proba (array-like): Predicted probabilities for the positive class.
        pos_label (int or str): The class considered as positive.
        dataset_name (str): Name of the dataset for the plot title.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. If None, a new figure is created.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
    if ax is None:
        plt.figure(figsize=(6, 5))
        plot_roc_curve(fpr, tpr, roc_auc, dataset_name)
    else:
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUROC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {dataset_name}')
        ax.legend(loc='lower right')


def display_roc_auc(y_train, y_val, train_pr_proba, val_pr_proba):
    """
    Displays the ROC-AUC curve for the training and validation sets.

    Args:
        y_train (array-like): Ground truth labels for the training set.
        y_val (array-like): Ground truth labels for the validation set.
        train_pr_proba (array-like): Predicted class probabilities for the training set.
        val_pr_proba (array-like): Predicted class probabilities for the validation set.

    Returns:
        None
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plot_roc_auc(y_train, train_pr_proba, 1, "Train", axes[0])
    plot_roc_auc(y_val, val_pr_proba, 1, "Validation", axes[1])
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def display_training_curves(history: dict):
    """
        Displays training and validation loss, as well as validation F1 score over epochs.

        Args:
            history (dict): Dictionary containing keys 'train_loss', 'val_loss', and 'val_f1',
                            each mapping to a list of metric values over training epochs.

        Raises:
            KeyError: If any of the required keys ('train_loss', 'val_loss', 'val_f1') are missing.

        Returns:
            None
    """

    required_keys = ["train_loss", "val_loss", "val_f1"]
    for key in required_keys:
        if key not in history:
            raise KeyError(f"history must contain '{key}'")

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(18, 6))

    ax[0].plot(epochs, history["train_loss"], 'go-', label='Training Loss')
    ax[0].plot(epochs, history["val_loss"], 'ro-', label='Validation Loss')
    ax[0].set_title('Training & Validation Loss')
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    ax[1].plot(epochs, history["val_f1"], 'bo-', label='Validation F1 Score')
    ax[1].set_title('Validation F1 Score')
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("F1 Score")
    ax[1].legend()

    plt.tight_layout()
    plt.show()


def display_all_confusion_matrices(train_y, val_y, test_y, train_pred, val_pred, test_pred):
    """
    Display normalized confusion matrices for train, validation, and test datasets.

    This function creates a single row of three side-by-side confusion matrices
    to compare model performance across datasets. The confusion matrices are
    normalized by true label counts.

    Args:
        train_y (array-like): True labels for the training set.
        val_y (array-like): True labels for the validation set.
        test_y (array-like): True labels for the test set.
        train_pred (array-like): Predicted labels for the training set.
        val_pred (array-like): Predicted labels for the validation set.
        test_pred (array-like): Predicted labels for the test set.

    Returns:
        None. Displays matplotlib figure with confusion matrices.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, y_true, y_p, title in zip(axes, [train_y, val_y, test_y], [train_pred, val_pred, test_pred],
                                      ["Train", "Validation", "Test"]):
        cm = confusion_matrix(y_true, y_p, normalize="true")
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{title} Confusion Matrix")
    plt.tight_layout()
    plt.show()


def display_all_roc_curves(train_y, val_y, test_y, train_probs, val_probs, test_probs):
    """
    Plot ROC curves for train, validation, and test datasets.

    This function plots the Receiver Operating Characteristic (ROC) curves for
    each dataset on a single graph. It computes and displays the Area Under
    the Curve (AUC) for each set to evaluate model discrimination performance.

    Args:
        train_y (array-like): True binary labels for the training set.
        val_y (array-like): True binary labels for the validation set.
        test_y (array-like): True binary labels for the test set.
        train_probs (array-like): Predicted probabilities for the training set.
        val_probs (array-like): Predicted probabilities for the validation set.
        test_probs (array-like): Predicted probabilities for the test set.

    Returns:
        None. Displays matplotlib figure with ROC curves.
    """
    plt.figure(figsize=(8, 6))
    for y_true, y_prob, name in zip([train_y, val_y, test_y], [train_probs, val_probs, test_probs],
                                    ["Train", "Validation", "Test"]):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
