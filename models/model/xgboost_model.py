import xgboost as xgb
from models.model.base_model import BaseModel


class XGBModel(BaseModel):
    """XGBoost model wrapper for binary classification.

    This class wraps `xgboost.XGBClassifier` and integrates it with the `BaseModel` interface.
    It supports parameter configuration and optional cross-validation setup.

    Inherits from:
        BaseModel: A custom base class for machine learning models with a unified interface.

    Attributes:
        params (dict): Parameters passed to the XGBoost classifier.
        kwargs (dict): Additional keyword arguments passed to BaseModel.
        model (xgb.XGBClassifier): The underlying XGBoost model instance.
    """

    def __init__(self, X_train, y_train, X_val, y_val, params=None, enable_cv=False, cv_params=None, **kwargs):
        """Initializes the XGBModel with training and validation data.

        Args:
            X_train (pd.DataFrame or np.ndarray): Training features.
            y_train (pd.Series or np.ndarray): Training labels.
            X_val (pd.DataFrame or np.ndarray): Validation features.
            y_val (pd.Series or np.ndarray): Validation labels.
            params (dict, optional): Parameters for the XGBClassifier. Defaults to None.
            enable_cv (bool, optional): Whether to enable cross-validation. Defaults to False.
            cv_params (dict, optional): Parameters for cross-validation if enabled.
            **kwargs: Additional keyword arguments passed to the BaseModel.
        """
        super().__init__(X_train, y_train, X_val, y_val, enable_cv=enable_cv, cv_params=cv_params, **kwargs)
        self.params = params if params else {}
        self.kwargs = kwargs
        self.model = xgb.XGBClassifier(**self.params)

    def fit(self):
        """Fits the XGBoost model to the training data.

        Returns:
            XGBModel: The instance of the trained model (self).
        """
        self.model.fit(self.X_train, self.y_train)
        return self
