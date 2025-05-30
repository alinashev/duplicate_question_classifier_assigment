from rich.console import Console

from models.utils.cross_validator import CrossValidator
from metrics.model_metrics import ModelEvaluation
from metrics.display import display_confusion_matrix, display_roc_auc, display_metrics, display_cross_validation_result


class BaseModel:
    """
    Base class for classical machine learning models with built-in explaining,
    visualization, and optional cross-validation support.

    This class is designed to be subclassed for specific model implementations
    (e.g., logistic regression, SVM, etc.).

    Args:
        X_train (np.ndarray or pd.DataFrame): Training features.
        y_train (np.ndarray or pd.Series): Training labels.
        X_val (np.ndarray or pd.DataFrame): Validation features.
        y_val (np.ndarray or pd.Series): Validation labels.
        pos_label (int, optional): Label considered as positive class. Defaults to 1.
        threshold (float, optional): Threshold for converting probabilities to binary predictions. Defaults to 0.5.
        enable_cv (bool, optional): Whether to perform cross-validation. Defaults to True.
        cv_params (dict, optional): Parameters to pass to the cross-validation logic. Defaults to None.
        **kwargs: Additional keyword arguments for extended flexibility.
    """

    def __init__(self, X_train, y_train, X_val, y_val, pos_label: int = 1, threshold: float = 0.5,
                 enable_cv: bool = True, cv_params: dict = None, **kwargs):
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.pos_label, self.threshold = pos_label, threshold
        self.model = None
        self.model_metrics = None
        self.enable_cv = enable_cv
        self.cv_params = cv_params or {}
        self.cv_results = None

    def fit(self):
        """
        Trains the model.

        This method should be overridden by child classes implementing a specific model.

        Raises:
            NotImplementedError: If the method is not implemented in the child class.
        """
        raise NotImplementedError("The fit() method must be implemented in a child class.")

    def predict(self):
        """
        Makes predictions and computes probabilities on both training and validation sets.
        Also updates explaining metrics.

        Returns:
            BaseModel: The current instance with updated predictions and metrics.
        """
        self.train_prediction = self.model.predict(self.X_train)
        self.val_prediction = self.model.predict(self.X_val)

        if hasattr(self.model, "predict_proba"):
            self.train_pr_proba = self.model.predict_proba(self.X_train)[:, self.pos_label]
            self.val_pr_proba = self.model.predict_proba(self.X_val)[:, self.pos_label]
        else:
            self.train_pr_proba = self.train_prediction
            self.val_pr_proba = self.val_prediction

        self.__get_metrix()
        return self

    def display_metrics(self):
        """
        Displays explaining metrics using `rich`, including cross-validation results if available.

        Returns:
            None
        """
        display_metrics(self.model_metrics)
        if self.cv_results:
            display_cross_validation_result(self.cv_results)

    def display_confusion_matrix(self):
        """
        Displays the confusion matrix for both training and validation sets.

        Returns:
            None
        """
        display_confusion_matrix(self.y_train, self.y_val, self.train_prediction, self.val_prediction)

    def display_roc_auc(self):
        """
        Displays ROC-AUC curves for both training and validation sets.

        Returns:
            None
        """
        display_roc_auc(self.y_train, self.y_val, self.train_pr_proba, self.val_pr_proba)

    def __get_metrix(self):
        """
        Computes and caches the explaining metrics for the current predictions.

        Returns:
            ModelEvaluation: An instance containing various classification metrics.
        """
        if self.model_metrics is None:
            self.model_metrics = ModelEvaluation(y_train=self.y_train,
                                                 y_val=self.y_val,
                                                 train_prediction=self.train_prediction,
                                                 val_prediction=self.val_prediction,
                                                 train_pr_proba=self.train_pr_proba,
                                                 val_pr_proba=self.val_pr_proba)
        return self.model_metrics

    def cross_validate(self):
        """
        Runs cross-validation if enabled, using the specified parameters.

        Returns:
            dict or None: A dictionary with cross-validation results, or None if cross-validation is disabled.
        """
        if self.enable_cv:
            print("Running cross-validation before training...")
            cv = CrossValidator(self.model, self.X_train, self.y_train, **self.cv_params)
            self.cv_results = cv.run()
        return self.cv_results
