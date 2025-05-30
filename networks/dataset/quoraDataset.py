from torch.utils.data import Dataset


class QuoraDataset(Dataset):
    """Custom Dataset for the Quora Question Pairs task.

    Each item contains a pair of tokenized questions and their binary label
    indicating whether the questions are duplicates.

    Attributes:
        q1_seqs (list or torch.Tensor): Tokenized sequences for question 1.
        q2_seqs (list or torch.Tensor): Tokenized sequences for question 2.
        labels (list or torch.Tensor): Binary labels (0 or 1).
    """

    def __init__(self, q1_seqs, q2_seqs, labels):
        """
        Initializes the dataset with tokenized question pairs and labels.

        Args:
            q1_seqs (list or torch.Tensor): Tokenized question 1 inputs.
            q2_seqs (list or torch.Tensor): Tokenized question 2 inputs.
            labels (list or torch.Tensor): Corresponding binary labels.
        """
        self.q1_seqs = q1_seqs
        self.q2_seqs = q2_seqs
        self.labels = labels

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieves the item at the specified index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            dict: A dictionary with keys:
                - 'question1': tokenized sequence for question 1,
                - 'question2': tokenized sequence for question 2,
                - 'label': binary label (0 or 1).
        """
        return {
            "question1": self.q1_seqs[idx],
            "question2": self.q2_seqs[idx],
            "label": self.labels[idx]
        }
