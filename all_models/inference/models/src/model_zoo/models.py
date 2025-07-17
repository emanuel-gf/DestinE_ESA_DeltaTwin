import segmentation_models_pytorch as smp
import torch.nn as nn

def define_model(
    name: str,
    encoder_name: str,
    out_channels: int = 3,
    in_channel: int = 3,
    encoder_weights: str = None,
    activation: str = None,
) -> nn.Module:
    """
    Define a segmentation model from segmentation_models_pytorch.

    Args:
        name (str): The name of the model (e.g., 'Unet', 'FPN', etc.).
        encoder_name (str): The name of the encoder (e.g., 'resnet34', 'resnet50', etc.).
        out_channels (int, optional): The number of output channels. Defaults to 3.
        in_channel (int, optional): The number of input channels. Defaults to 3.
        encoder_weights (str, optional): The pre-trained weights for the encoder. Defaults to None.
        activation (str, optional): The activation function to apply after the model. Defaults to None.

    Returns:
        nn.Module: The defined segmentation model.

    Raises:
        ValueError: If the model name is not found in segmentation_models_pytorch.
    """
    try:
        ModelClass = getattr(smp, name)
        model = ModelClass(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channel,
            classes=out_channels,
        )

        if activation:
            activation_layer = {
                "relu": nn.ReLU(),
                "sigmoid": nn.Sigmoid()
            }.get(activation.lower())

            if activation_layer:
                model = nn.Sequential(model, activation_layer)
            else:
                raise ValueError(f"Unsupported activation function: {activation}")

        return model
    except AttributeError:
        raise ValueError(f"Model '{name}' not found in segmentation_models_pytorch. Available models: {dir(smp)}")
