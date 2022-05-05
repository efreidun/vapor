"""Module that defines the model classes."""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


def tran_conv_layer(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    activation: Optional[nn.Module] = None,
) -> nn.Sequential:
    """Create a transposed convolutional layer.
    Args:
        in_channels: number of channels at input
        out_channels: number of channels at output
        kernel_size: size of the kernel. if integer, value is used for both axes
        stride: stride of convolution. if integer, value is used for both axes
        padding: padding used for convolution. if integer, value is used for both axes
        activation: activation to be applied after transposed convolution operation
    Returns:
        transposed convolutional layer object
    """
    layers = [
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
    ]
    if activation is not None:
        layers.append(activation)
    return nn.Sequential(*layers)


def conv_layer(
    in_features: int,
    out_features: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    activation: Optional[nn.Module] = None,
) -> nn.Module:
    """Sequence of a convolutional layer with optional activation.

    Args:
        in_features: number of input features
        out_features: number of output features
        kernel_size: kernel size
        stride: convolution operation stride
        padding: padding size of the input before convolution operation
        activation: optional activation after the layer

    Returns:
        sequential_layer: created sequential layer
    """
    layers = [
        nn.Conv2d(
            in_features,
            out_features,
            kernel_size=kernel_size,
            padding=padding,  # (kernel_size - 1) // 2,
            stride=stride,
        )
    ]
    if activation is not None:
        layers.append(activation)
    sequential_layer = nn.Sequential(*layers)

    return sequential_layer


def linear_layer(
    in_features: int, out_features: int, activation: Optional[nn.Module] = None
) -> nn.Module:
    """Sequence of a linear layer with optional activation.

    Args:
        in_features: number of input connections to the layer
        out_features: number of nodes in the layer
        activation: optional activation after linear layer

    Returns:
        sequential_layer: created sequential layer
    """
    layers = [nn.Linear(in_features, out_features)]
    if activation is not None:
        layers.append(activation)
    sequential_layer = nn.Sequential(*layers)

    return sequential_layer


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self._latent_dim = latent_dim
        self.conv1 = conv_layer(3, 32, 4, 2, 1, nn.ReLU())  # (32, 32, 32)
        self.conv2 = conv_layer(32, 32, 4, 2, 1, nn.ReLU())  # (32, 16, 16)
        self.conv3 = conv_layer(32, 64, 4, 2, 1, nn.ReLU())  # (64, 8, 8)
        self.conv4 = conv_layer(64, 64, 4, 2, 1, nn.ReLU())  # (64, 4, 4)
        self.conv5 = conv_layer(64, 256, 4, 1, 0, nn.ReLU())  # (256, 1, 1)
        self.linear = linear_layer(256, 128, nn.ReLU())  # (128,)
        self.mu = linear_layer(128, latent_dim)  # (latent_dim,)
        self.logvar = linear_layer(128, latent_dim)  # (latent_dim,)

    def forward(self, batch):
        batch = self.conv1(batch)
        batch = self.conv2(batch)
        batch = self.conv3(batch)
        batch = self.conv4(batch)
        batch = self.conv5(batch).reshape(-1, 256)
        batch = self.linear(batch)
        mu = self.mu(batch)
        logvar = self.mu(batch)

        return mu, logvar

    def get_latent_dim(self) -> int:
        """Retrieve latent dimension of the model."""
        return self._latent_dim


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self._latent_dim = latent_dim
        self.linear = linear_layer(latent_dim, 256, nn.ReLU())  # (256,)
        self.tran_conv1 = tran_conv_layer(256, 64, 4, 1, 0, nn.ReLU())  # (64, 4, 4)
        self.tran_conv2 = tran_conv_layer(64, 64, 4, 2, 1, nn.ReLU())  # (64, 8, 8)
        self.tran_conv3 = tran_conv_layer(64, 32, 4, 2, 1, nn.ReLU())  # (32, 16, 16)
        self.tran_conv4 = tran_conv_layer(32, 32, 4, 2, 1, nn.ReLU())  # (32, 32, 32)
        self.tran_conv5 = tran_conv_layer(32, 3, 4, 2, 1)  # (3, 64, 64)

    def forward(self, batch):
        batch = self.linear(batch).reshape(-1, 256, 1, 1)
        batch = self.tran_conv1(batch)
        batch = self.tran_conv2(batch)
        batch = self.tran_conv3(batch)
        batch = self.tran_conv4(batch)
        batch = self.tran_conv5(batch)

        return batch

    def get_latent_dim(self) -> int:
        """Retrieve latent dimension of the model."""
        return self._latent_dim


class PoseMap(nn.Module):
    """Class for pose latent to SE(3) map."""

    def __init__(self, latent_dim: int) -> None:
        """Initialize the map.

        Args:
            latent_dim: number of latent dimensions
        """
        super().__init__()
        self._latent_dim = latent_dim
        self.fc1 = linear_layer(latent_dim, 128, nn.ReLU())
        self.fc2 = linear_layer(128, 256, nn.ReLU())
        self.fc3 = linear_layer(256, 256, nn.ReLU())
        self.fc4 = linear_layer(256, 256, nn.ReLU())
        self.fc5 = linear_layer(256, 256, nn.ReLU())
        self.fc6 = linear_layer(256, 256, nn.ReLU())
        self.fc7 = linear_layer(256, 256, nn.ReLU())
        self.fc8 = linear_layer(256, 256, nn.ReLU())
        self.fc9 = linear_layer(256, 256, nn.ReLU())
        self.fc10 = linear_layer(256, 256, nn.ReLU())
        self.fc11 = linear_layer(256, 256, nn.ReLU())
        self.fc12 = linear_layer(256, 128, nn.ReLU())
        self.fc_tvec = linear_layer(128, 3, nn.Sigmoid())
        self.fc_rvec = linear_layer(128, 6)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the map.

        Args:
            z: latent vector, shape (N, latent_dim)

        Returns:
            tvec: translation vector, shape (N, 3)
            rvec: orientation continous 6D representation, shape (N, 6)
        """
        batch = self.fc1(z)
        batch = self.fc2(batch)
        batch = self.fc3(batch)
        batch = self.fc4(batch)
        batch = self.fc5(batch)
        batch = self.fc6(batch)
        batch = self.fc7(batch)
        batch = self.fc8(batch)
        batch = self.fc9(batch)
        batch = self.fc10(batch)
        batch = self.fc11(batch)
        batch = self.fc12(batch)
        tvec = self.fc_tvec(batch)
        rvec = self.fc_rvec(batch)

        return tvec, rvec

    def get_latent_dim(self) -> int:
        """Retrieve latent dimension of the model."""
        return self._latent_dim
