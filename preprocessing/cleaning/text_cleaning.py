import re
import string

from bs4 import BeautifulSoup
from nltk import SnowballStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Cleans raw text by applying:
    - lowercasing
    - HTML removal
    - URL and email removal
    - source removal (e.g., 'Reuters', 'via')
    - punctuation and non-ASCII removal
    - whitespace normalization

    Args:
        text (str): Raw input text.

    Returns:
        str: Cleaned text string.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', ' ', text)
    text = re.sub(r'\b(reuters|via)\b\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def remove_stopwords(text: str) -> list:
    """Removes English stopwords from tokenized input text.

    Args:
        text (str): The cleaned text.

    Returns:
        list: A list of tokens with stopwords removed.
    """
    stop_words = set(stopwords.words("english"))
    if isinstance(text, str):
        tokens = text.lower().split()
        return [word for word in tokens if word not in stop_words]
    return []


def stem_tokens(tokens: list) -> list:
    """Applies stemming to a list of tokens.

    Args:
        tokens (list): A list of word tokens.

    Returns:
        list: A list of stemmed tokens.
    """
    return [stemmer.stem(token) for token in tokens]


def lemmatize_tokens(tokens: list) -> list:
    """Applies lemmatization to a list of tokens.

    Args:
        tokens (list): A list of word tokens.

    Returns:
        list: A list of lemmatized tokens.
    """
    return [lemmatizer.lemmatize(token) for token in tokens]


def process_text(text: str) -> list:
    """Performs full preprocessing pipeline: cleaning, stopword removal, and lemmatization.

    This function is typically used before applying classical ML models such as
    TF-IDF + Logistic Regression.

    Args:
        text (str): The raw input text.

    Returns:
        list: A list of lemmatized tokens.
    """
    cleaned = clean_text(text)
    tokens = remove_stopwords(cleaned)
    lemmatized = lemmatize_tokens(tokens)
    return lemmatized


def process_text_tokens(text: str) -> list:
    """Cleans the text and returns a list of tokens (no stopword removal or lemmatization).

    This can be used for models that expect raw tokenized inputs.

    Args:
        text (str): The raw input text.

    Returns:
        list: A list of cleaned word tokens.
    """
    cleaned = clean_text(text)
    return cleaned.split()
