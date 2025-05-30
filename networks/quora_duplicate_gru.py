import torch
import torch.nn as nn


class QuoraDuplicateGRU(nn.Module):
    """
    Bidirectional GRU model for duplicate question classification (Quora Question Pairs).

    The model encodes two input questions using a shared BiGRU encoder. Their representations
    are combined via concatenation of standard vector operations, and passed through
    a fully connected layer to predict the probability of duplication.

    """

    def __init__(self, embedding_matrix, hidden_size=128, dropout=0.3):
        """
        Initializes the QuoraDuplicateGRU model.

        Args:
            embedding_matrix (torch.Tensor): Pretrained embedding weights (vocab_size x embedding_dim).
            hidden_size (int, optional): GRU hidden state size (per direction). Defaults to 128.
            dropout (float, optional): Dropout probability. Defaults to 0.3.
        """
        super().__init__()
        vocab_size, embedding_dim = embedding_matrix.shape

        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix.detach().clone(),
            freeze=False,
            padding_idx=0
        )

        self.q_encoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 8, 1)

    def encode_question(self, x):
        """
        Encodes a question using BiGRU.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Encoded question representation of shape (batch_size, hidden_size * 2).
        """
        embedded = self.embedding(x)
        _, hidden = self.q_encoder(embedded)
        hidden = torch.cat((hidden[0], hidden[1]), dim=1)
        return hidden

    def forward(self, q1_input, q2_input):
        """
        Forward pass for a pair of questions.

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
