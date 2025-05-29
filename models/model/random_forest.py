from sklearn.ensemble import RandomForestClassifier

from models.model.base_model import BaseModel


class RandomForestModel(BaseModel):
    """Random Forest model wrapper for binary classification.

    This class wraps scikit-learn's `RandomForestClassifier` and integrates it with the `BaseModel` interface.
    It supports optional cross-validation and allows passing additional model parameters via kwargs.

    Inherits from:
        BaseModel: A custom base class for machine learning models with a unified interface.

    Attributes:
        model (RandomForestClassifier): The underlying scikit-learn random forest classifier.
    """

    def __init__(self, X_train, y_train, X_val, y_val, enable_cv=False, cv_params=None, **kwargs):
        """Initializes the RandomForestModel with training and validation data.

        Args:
            X_train (pd.DataFrame or np.ndarray): Training features.
            y_train (pd.Series or np.ndarray): Training labels.
            X_val (pd.DataFrame or np.ndarray): Validation features.
            y_val (pd.Series or np.ndarray): Validation labels.
            enable_cv (bool, optional): Whether to perform cross-validation. Defaults to False.
            cv_params (dict, optional): Parameters for cross-validation if enabled.
            **kwargs: Additional keyword arguments passed to `RandomForestClassifier`.
        """
        super().__init__(X_train, y_train, X_val, y_val, enable_cv=enable_cv, cv_params=cv_params, **kwargs)

        self.model = RandomForestClassifier(
            random_state=42,
            **kwargs
        )

    def fit(self):
        """Fits the random forest model on the training data.

        If cross-validation is enabled, it is run before fitting.

        Returns:
            RandomForestModel: The instance of the trained model (self).
        """
        self.cross_validate()
        self.model.fit(self.X_train, self.y_train)
        return self
