from torch import nn


class DuplicateTextClassifier(nn.Module):
    """
    Transformer-based classifier for detecting duplicate text pairs.

    This model wraps a pre-trained BERT-like model and adds a feedforward
    classification head. It uses the [CLS] token representation for classification.

    Args:
        bert_model (PreTrainedModel): A pre-trained BERT-like model from HuggingFace Transformers.
        hidden_dim (int, optional): Hidden layer size in the classifier head. Defaults to 128.
        dropout_prob (float, optional): Dropout probability for regularization. Defaults to 0.1.
        num_classes (int, optional): Number of output classes. Defaults to 2 (binary classification).

    Attributes:
        bert (PreTrainedModel): Backbone transformer model.
        dropout (nn.Dropout): Dropout layer.
        classifier (nn.Sequential): Fully connected classifier head.
    """

    def __init__(self, bert_model, hidden_dim=128, dropout_prob=0.1, num_classes=2):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Sequential(
            nn.Linear(bert_model.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        """
            Perform a forward pass through the model.

            Args:
                input_ids (torch.Tensor): Tensor of token IDs with shape (batch_size, seq_len).
                attention_mask (torch.Tensor, optional): Mask tensor indicating valid tokens.
                token_type_ids (torch.Tensor, optional): Segment token indicators for sentence pairs.
                labels (torch.Tensor, optional): Ground truth labels. If provided, loss will be computed.

            Returns:
                dict: A dictionary containing:
                    - logits (torch.Tensor): Raw predictions before softmax.
                    - loss (torch.Tensor, optional): Cross-entropy loss if labels are provided.
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        cls_output = outputs.pooler_output
        x = self.dropout(cls_output)
        logits = self.classifier(x)

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
