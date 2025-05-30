import torch
import torch.nn as nn


class QuoraLstmDuplicateClassifier(nn.Module):
    """
    BiLSTM-based model for detecting duplicate question pairs from the Quora dataset.

    This model processes question1 and question2 independently using a shared embedding layer
    and separate BiLSTM encoders, then concatenates their representations to classify if the pair is a duplicate.

    """

    def __init__(self, embedding_matrix, hidden_size=128, dropout=0.3):
        super(QuoraLstmDuplicateClassifier, self).__init__()

        vocab_size, embedding_dim = embedding_matrix.shape

        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, padding_idx=0, freeze=False
        )

        self.q1_encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )

        self.q2_encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 4, 1)

    def _encode_question(self, x, encoder):
        """
        Encodes a question using the given BiLSTM encoder.

        Args:
            x (torch.Tensor): Input tensor of token ids (batch_size, seq_len).
            encoder (nn.LSTM): The LSTM encoder to use.

        Returns:
            torch.Tensor: (batch_size, 2 * hidden_size)
        """
        embedded = self.embedding(x)
        _, (hidden, _) = encoder(embedded)
        return torch.cat((hidden[0], hidden[1]), dim=1)

    def forward(self, q1_input, q2_input):
        """
        Forward pass through the model.

        Args:
            q1_input (torch.Tensor): Tensor of shape (batch_size, seq_len) for question1.
            q2_input (torch.Tensor): Tensor of shape (batch_size, seq_len) for question2.

        Returns:
            torch.Tensor: Output logits of shape (batch_size,).
        """
        q1_repr = self._encode_question(q1_input, self.q1_encoder)
        q2_repr = self._encode_question(q2_input, self.q2_encoder)

        combined = torch.cat((q1_repr, q2_repr), dim=1)
        x = self.dropout(combined)
        return self.fc(x).squeeze(1)
