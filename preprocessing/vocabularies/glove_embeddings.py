import os
import zipfile
import urllib.request
import numpy as np


def prepare_glove_embeddings(
    dims: str = "300",
    target_dir: str = ".",
    source: str = "url",
    local_zip_path: str = None
) -> str:
    """
    Prepares GloVe embedding file by checking for existence, optionally downloading and extracting from ZIP.

    Args:
        dims (str): Dimensionality of the GloVe vectors (e.g., "50", "100", "200", "300").
        target_dir (str): Directory to save or search for GloVe files.
        source (str): Source type, either "url" (to download from the internet) or "local".
        local_zip_path (str, optional): Path to a local GloVe ZIP file (if available).

    Returns:
        str: Path to the extracted GloVe `.txt` file.

    Raises:
        FileNotFoundError: If files are not found in local mode.
        ValueError: If an unsupported source type is provided.
    """
    txt_filename = f"glove.6B.{dims}d.txt"
    txt_path = os.path.join(target_dir, txt_filename)
    os.makedirs(target_dir, exist_ok=True)

    zip_filename = "glove.6B.zip"
    zip_path = local_zip_path or os.path.join(target_dir, zip_filename)

    if os.path.exists(txt_path):
        print(f"The file was found: {txt_path}")
        return txt_path

    if os.path.exists(zip_path):
        print(f"Unpacking: {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(path=target_dir)
        return txt_path

    if source == "local":
        raise FileNotFoundError("No .txt or .zip found in local mode..")

    if source == "url":
        url = "http://nlp.stanford.edu/data/glove.6B.zip"
        print(f"Loading from {url}...")
        urllib.request.urlretrieve(url, zip_path)

        print(f"Unpacking in {target_dir}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(path=target_dir)

        return txt_path

    raise ValueError(f"Unknown value for source: {source}")



def load_glove_embeddings(glove_path: str, embedding_dim: int) -> dict:
    """
    Loads GloVe word embeddings from a text file.

    Args:
        glove_path (str): Path to the GloVe `.txt` file.
        embedding_dim (int): Expected dimensionality of each embedding vector.

    Returns:
        dict: A dictionary mapping words to their embedding vectors as numpy arrays.
    """
    embeddings = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            if len(vector) == embedding_dim:
                embeddings[word] = vector
    return embeddings

