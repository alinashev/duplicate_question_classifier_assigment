from sklearn.metrics import f1_score, roc_curve, auc, log_loss


class Metric:
    """
    Represents a single explaining metric for a model.

    Attributes:
        name (str): The name of the metric (e.g., 'accuracy', 'f1_score').
        train_value (float): The value of the metric for the training data.
        val_value (float): The value of the metric for the validation data.
    """

    def __init__(self, name: str, train_value: float, val_value: float):
        """
        Initializes a Metric instance.

        Args:
            name (str): The name of the metric.
            train_value (float): The value of the metric for the training data.
            val_value (float): The value of the metric for the validation data.
        """
        self.name = name
        self.train_value = train_value
        self.val_value = val_value


class ModelEvaluation:
    """
    A class for calculating and storing performance metrics for machine learning models.

    Attributes:
        y_train (array-like): The true labels for the training data.
        y_val (array-like): The true labels for the validation data.
        pos_label (int): The label for the positive class (default is 1).
        threshold (float): The threshold used to classify predictions as the positive class (default is 0.5).
        metrics (dict): A dictionary to store computed metrics.
        train_prediction (array-like): The predicted labels for the training data.
        val_prediction (array-like): The predicted labels for the validation data.
        train_pr_proba (array-like): The predicted probabilities for the positive class for the training data.
        val_pr_proba (array-like): The predicted probabilities for the positive class for the validation data.
    """

    def __init__(self,  y_train=None, y_val=None, pos_label=1, threshold=0.5,
                 train_prediction=None, val_prediction=None, train_pr_proba=None, val_pr_proba=None):
        """
        Initializes the ModelMetrics instance with true labels, predicted labels, probabilities, and other settings.

        Args:
            y_train (array-like): The true labels for the training data.
            y_val (array-like): The true labels for the validation data.
            pos_label (int): The label for the positive class (default is 1).
            threshold (float): The threshold used to classify predictions as positive (default is 0.5).
            train_prediction (array-like): The predicted labels for the training data.
            val_prediction (array-like): The predicted labels for the validation data.
            train_pr_proba (array-like): The predicted probabilities for the training data.
            val_pr_proba (array-like): The predicted probabilities for the validation data.
        """

        self.y_train = y_train
        self.y_val = y_val
        self.pos_label = pos_label
        self.threshold = threshold
        self.metrics = []

        self.train_prediction = train_prediction
        self.val_prediction = val_prediction
        self.train_pr_proba = train_pr_proba
        self.val_pr_proba = val_pr_proba


    def f1_score_metric(self):
        """
        Computes the F1-score metric.

        Returns:
            Metric: The computed F1-score metric.
        """
        f1_train = f1_score(self.y_train, self.train_prediction)
        f1_val = f1_score(self.y_val, self.val_prediction)
        metric = Metric("f1_score", f1_train, f1_val)
        self.metrics.append(metric)
        return metric


    def roc_auc(self):
        """
        Computes the ROC-AUC metric.

        Returns:
            Metric: The computed ROC-AUC metric.
        """
        fpr_t, tpr_t, _ = roc_curve(self.y_train, self.train_pr_proba)
        fpr_v, tpr_v, _ = roc_curve(self.y_val, self.val_pr_proba)

        roc_auc_train = auc(fpr_t, tpr_t)
        roc_auc_valid = auc(fpr_v, tpr_v)

        metric = Metric("roc_auc", roc_auc_train, roc_auc_valid)
        self.metrics.append(metric)
        return metric

    def log_loss_metric(self):
        """
        Computes log loss (binary cross-entropy) for training and validation data, if probabilities are available.

        Returns:
            Metric: An object containing log loss for train and validation sets. May contain `None` if data is missing.
        """

        log_train = log_val = None

        if self.y_train is not None and self.train_pr_proba is not None:
            log_train = log_loss(self.y_train, self.train_pr_proba)

        if self.y_val is not None and self.val_pr_proba is not None:
            log_val = log_loss(self.y_val, self.val_pr_proba)

        metric = Metric("log_loss", log_train, log_val)
        self.metrics.append(metric)
        return metric

    def compute_all_metrics(self):
        """
        Computes and stores all enabled metrics as per the configuration.

        Returns:
            list: A list of computed Metric objects.
        """
        self.metrics = []
        metric_functions = {
            "f1_score": self.f1_score_metric,
            "roc_auc": self.roc_auc,
        }

        for metric_name, func in metric_functions.items():
            func()
        return self.metrics
