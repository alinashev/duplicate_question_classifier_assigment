from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
import numpy as np

class CrossValidator:
    """
    A class for performing cross-validation using different strategies and explaining metrics.

    Attributes:
        model (object): The machine learning model to be validated.
        X (array-like): Feature matrix for cross-validation.
        y (array-like): Target labels.
        cv (int): Number of cross-validation folds (default: 5).
        scoring (list[str]): List of scoring metrics to evaluate (default: ["roc_auc"]).
        strategy (str): Cross-validation strategy - "stratified", "kfold" (default: "stratified").
    """

    def __init__(self, model, X, y, cv: int = 5, scoring: list = ["f1"], strategy: str = "kfold"):
        """
        Initializes the CrossValidator instance with a model, dataset, and cross-validation settings.

        Args:
            model (object): The machine learning model to validate.
            X (array-like): Feature matrix for cross-validation.
            y (array-like): Target labels.
            cv (int, optional): Number of folds for cross-validation. Defaults to 5.
            scoring (list[str] or str, optional): Scoring metrics to evaluate. Defaults to ["roc_auc"].
            strategy (str, optional): Cross-validation strategy - "stratified" or "kfold". Defaults to "stratified".
        """
        self.model = model
        self.X = X
        self.y = y
        self.cv = cv
        self.scoring = scoring if isinstance(scoring, list) else [scoring]
        self.strategy = strategy

    def get_cv_strategy(self):
        """
        Returns the appropriate cross-validation strategy based on the specified type.

        Returns:
            object: An instance of a cross-validation splitting strategy.

        Raises:
            ValueError: If an invalid strategy is specified.
        """
        if self.strategy == "stratified":
            return StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
        elif self.strategy == "kfold":
            return KFold(n_splits=self.cv, shuffle=True, random_state=42)
        else:
            raise ValueError("Invalid strategy! Choose from: 'stratified', 'kfold'.")

    def run(self) -> dict:
        """
        Runs cross-validation and computes the average score and standard deviation for each metric.

        Returns:
            dict: A dictionary containing the mean and standard deviation of cross-validation scores per metric.
        """
        cv_strategy = self.get_cv_strategy()
        results = {}

        for metric in self.scoring:
            scores = cross_val_score(self.model, self.X, self.y, cv=cv_strategy, scoring=metric)
            results[metric] = {"mean": np.mean(scores), "std": np.std(scores)}
            print(f"Cross-validation {metric}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

        return results