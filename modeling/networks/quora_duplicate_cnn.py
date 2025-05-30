import torch
import torch.nn as nn


class QuoraDuplicateCNN(nn.Module):
    """
    Convolutional Neural Network for Quora Question Pairs classification.

    This model encodes two questions using a shared CNN encoder and compares their
    representations via concatenation of vector operations (abs difference, elementwise product),
    followed by a fully connected layer for binary classification.
    """

    def __init__(self, embedding_matrix, num_filters=128, kernel_size=3, dropout=0.3):
        """Initializes the CNN model.
        Args:
            embedding_matrix (torch.Tensor): Pretrained embedding weights (vocab_size x embedding_dim).
            num_filters (int, optional): Number of convolutional filters. Defaults to 128.
            kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
            dropout (float, optional): Dropout probability. Defaults to 0.3.
        """
        super().__init__()
        vocab_size, embedding_dim = embedding_matrix.shape

        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix.detach().clone(),
            freeze=False,
            padding_idx=0
        )

        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=1
        )

        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * 4, 1)

    def encode_question(self, x):
        """
        Encodes a question sequence via embedding + convolution + pooling.

        Args:
            x (torch.Tensor): Tokenized question input of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Encoded vector representation of shape (batch_size, num_filters).
        """
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        embedded = embedded.permute(0, 2, 1)  # (batch_size, embed_dim, seq_len)
        conv_out = self.relu(self.conv(embedded))  # (batch_size, num_filters, seq_len)
        pooled = self.pool(conv_out).squeeze(-1)  # (batch_size, num_filters)
        return pooled

    def forward(self, q1_input, q2_input):
        """
        Forward pass for the pair of questions.

        Args:
            q1_input (torch.Tensor): Tokenized input for question 1, shape (batch_size, seq_len).
            q2_input (torch.Tensor): Tokenized input for question 2, shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Logits for binary classification, shape (batch_size).
        """
        q1_repr = self.encode_question(q1_input)
        q2_repr = self.encode_question(q2_input)

        combined = torch.cat(
            [q1_repr, q2_repr, torch.abs(q1_repr - q2_repr), q1_repr * q2_repr],
            dim=1
        )
        x = self.dropout(combined)
        return self.fc(x).squeeze(1)
