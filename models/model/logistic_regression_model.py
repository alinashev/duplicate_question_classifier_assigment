from sklearn.linear_model import LogisticRegression
from models.model.base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    """Logistic Regression model wrapper for binary classification.

    This class wraps scikit-learn's `LogisticRegression` and integrates it with the `BaseModel` interface.
    It supports an optional enricher (e.g., a feature transformer), custom threshold, and positive class label.

    Inherits from:
        BaseModel: A custom base class for machine learning models with a unified interface.

    Attributes:
        enricher (object or None): Optional transformer with `fit_transform` and `transform` methods.
        model (LogisticRegression): The underlying scikit-learn logistic regression model.
    """

    def __init__(self, X_train, y_train, X_val, y_val, pos_label=1, threshold=0.5, enricher=None, **kwargs):
        """Initializes the LogisticRegressionModel with training and validation data.

        Args:
            X_train (pd.DataFrame or np.ndarray): Training features.
            y_train (pd.Series or np.ndarray): Training labels.
            X_val (pd.DataFrame or np.ndarray): Validation features.
            y_val (pd.Series or np.ndarray): Validation labels.
            pos_label (int, optional): The label considered as positive class. Defaults to 1.
            threshold (float, optional): Threshold for converting probabilities to class predictions. Defaults to 0.5.
            enricher (object, optional): Optional feature transformer with `fit_transform()` and `transform()` methods.
            **kwargs: Additional keyword arguments passed to `LogisticRegression`.
        """
        self.enricher = enricher
        X_train_transformed = self.enricher.fit_transform(X_train) if self.enricher else X_train
        X_val_transformed = self.enricher.transform(X_val) if self.enricher else X_val

        super().__init__(X_train_transformed, y_train, X_val_transformed, y_val, pos_label, threshold, **kwargs)
        self.model = LogisticRegression(solver='liblinear', **kwargs)

    def fit(self):
        """Fits the logistic regression model to the training data.

        Returns:
            LogisticRegressionModel: The instance of the trained model (self).
        """
        self.cross_validate()
        self.model.fit(self.X_train, self.y_train)
        return self
