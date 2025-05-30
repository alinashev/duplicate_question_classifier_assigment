from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

class TfidfTextVectorizer(BaseEstimator, TransformerMixin):
    """
        Custom TF-IDF vectorizer for use in sklearn pipelines. Applies TfidfVectorizer to a specified
        text column in a Pandas DataFrame.

        Args:
            text_column (str, optional): Name of the column containing the text. Defaults to "full_text".
            tokenizer (callable, optional): Custom tokenizer or preprocessing function. Defaults to None.
            max_features (int, optional): Maximum number of features (vocabulary size). Defaults to 1000.
            min_df (int, optional): Minimum number of documents a term must appear in to be included. Defaults to 2.
    """
    def __init__(self, text_column="full_text", tokenizer=None, max_features=1000, min_df=2):
        self.text_column = text_column
        self.tokenizer = tokenizer
        self.max_features = max_features
        self.min_df = min_df
        self.vectorizer = TfidfVectorizer(
            tokenizer=self.tokenizer,
            max_features=self.max_features,
            min_df=self.min_df
        )

    def fit(self, X):
        """
        Fits the internal TfidfVectorizer on the specified text column.

        Args:
            X (pd.DataFrame): Input DataFrame containing the text column.
            y: Ignored. Present for compatibility with sklearn pipelines.

        Returns:
            TfidfTextVectorizer: The fitted instance of the transformer.
        """
        texts = X[self.text_column].astype(str).tolist()
        self.vectorizer.fit(texts)
        return self

    def transform(self, X):
        """
        Transforms the text column into a sparse TF-IDF matrix.

        Args:
            X (pd.DataFrame): Input DataFrame containing the text column.

        Returns:
            scipy.sparse.csr_matrix: Sparse matrix of shape (n_samples, n_features).
        """
        texts = X[self.text_column].astype(str).tolist()
        return self.vectorizer.transform(texts)
