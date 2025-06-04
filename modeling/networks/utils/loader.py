import torch
import os
from typing import Union
from torch import nn


def save_model_state(model, path: str = "fake_news_model.pt") -> None:
    """
    Saves the state dictionary of a PyTorch model to a file.

    Args:
        model: The PyTorch model whose state_dict is to be saved.
        path (str): The path where the state_dict will be saved.
                    Defaults to "fake_news_model.pt".

    Returns:
        None
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model state_dict saved to: {os.path.abspath(path)}")


def load_model_state(
        model_class: nn.Module,
        embedding_tensor: torch.Tensor,
        path: str,
        device: Union[str, torch.device] = "cpu"
) -> nn.Module:
    """
    Loads the state dictionary into a PyTorch model and returns the model
    in evaluation mode.

    Args:
        model_class (nn.Module): The model class to instantiate (should accept the embedding tensor as an argument).
        embedding_tensor (torch.Tensor): The embedding tensor to be passed to the model.
        path (str): The path to the saved state_dict file.
        device (Union[str, torch.device]): The device to load the model onto. Defaults to "cpu".

    Returns:
        nn.Module: The model loaded with the state_dict and set to evaluation mode.

    Raises:
        AssertionError: If any model parameter is not on the specified device.
    """
    device = torch.device(device)

    embedding_tensor = embedding_tensor.to(device)

    model = model_class(embedding_tensor).to(device)

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    for name, param in model.named_parameters():
        if param.device.type != device.type:
            raise AssertionError(
                f"Parameter `{name}` is not on {device.type}, but on {param.device.type}"
            )

    print(f"Model loaded from {os.path.abspath(path)} to {device}")
    return model
