"""Module that defines the model classes."""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torchvision import models


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
    batch_norm: bool = False,
) -> nn.Module:
    """Sequence of a convolutional layer with optional activation.

    Args:
        in_features: number of input features
        out_features: number of output features
        kernel_size: kernel size
        stride: convolution operation stride
        padding: padding size of the input before convolution operation
        activation: optional activation after the layer
        batch_norm: if True, BatchNorm2d layers are added

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
        ),
    ]
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_features))
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
    def __init__(self, latent_dim, backbone="resnet34"):
        super().__init__()
        self._latent_dim = latent_dim
        if backbone == "resnet18":
            self.model = models.resnet18(pretrained=True)
        elif backbone == "resnet34":
            self.model = models.resnet34(pretrained=True)
        else:
            raise NotImplementedError("Invalid backbone.")
        fe_out_planes = self.model.fc.in_features
        self.model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.model.fc = linear_layer(fe_out_planes, 2048, nn.ReLU())
        self.mu = linear_layer(2048, latent_dim)  # (latent_dim,)
        self.logvar = linear_layer(2048, latent_dim)  # (latent_dim,)

    def forward(self, batch):
        batch = self.model(batch)
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

    def __init__(self, latent_dim: int, depth: int, breadth: int) -> None:
        """Initialize the map.

        Args:
            latent_dim: number of latent dimensions
            depth: number of hidden layers
            breadth: number of neurons in hidden layers
        """
        super().__init__()
        self._latent_dim = latent_dim
        self.input_layer = linear_layer(latent_dim, breadth, nn.ReLU())
        self.layers = nn.ModuleList(
            [linear_layer(breadth, breadth, nn.ReLU()) for _ in range(depth)]
        )
        self.fc_tvec = linear_layer(breadth, 3, nn.Sigmoid())
        self.fc_rvec = linear_layer(breadth, 6)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the map.

        Args:
            z: latent vector, shape (N, latent_dim)

        Returns:
            tvec: translation vector, shape (N, 3)
            rvec: orientation continous 6D representation, shape (N, 6)
        """
        batch = self.input_layer(z)
        for layer in self.layers:
            batch = layer(batch)
        tvec = self.fc_tvec(batch)
        rvec = self.fc_rvec(batch)

        return tvec, rvec

    def get_latent_dim(self) -> int:
        """Retrieve latent dimension of the model."""
        return self._latent_dim
