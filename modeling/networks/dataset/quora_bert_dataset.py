import torch


class QuestionDataset(torch.utils.data.Dataset):
    """
    Custom PyTorch Dataset for processing pairs of questions (e.g., for duplicate detection).

    This dataset takes two lists of questions, their corresponding labels, and a tokenizer.
    It applies tokenization with padding and truncation to generate inputs suitable for
    transformer-based models like BERT.

    Args:
        questions1 (List[str]): First set of questions (e.g., question1 in a pair).
        questions2 (List[str]): Second set of questions (e.g., question2 in a pair).
        labels (List[int]): Binary labels indicating relationship (e.g., duplicate or not).
        tokenizer (PreTrainedTokenizer): Tokenizer from HuggingFace's transformers library.
        max_length (int, optional): Maximum sequence length after tokenization. Defaults to 128.

    Attributes:
        questions1 (List[str])
        questions2 (List[str])
        labels (List[int])
        tokenizer (PreTrainedTokenizer)
        max_length (int)
    """

    def __init__(self, questions1, questions2, labels, tokenizer, max_length=128):
        self.questions1 = questions1
        self.questions2 = questions2
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """
        Return the number of examples in the dataset.

        Returns:
            int: Total number of question pairs.
        """
        return len(self.questions1)

    def __getitem__(self, idx):
        """
        Get tokenized inputs and label for a specific index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing:
                - input_ids (Tensor): Token IDs.
                - attention_mask (Tensor): Attention mask.
                - token_type_ids (Tensor): Segment token indicators.
                - labels (Tensor): Label for the question pair.
        """
        encoding = self.tokenizer(
            self.questions1[idx],
            self.questions2[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            return_overflowing_tokens=False
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'token_type_ids': encoding['token_type_ids'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
        }