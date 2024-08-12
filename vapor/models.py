"""Module that defines the model classes."""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from normflows.flows import Planar, Radial
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
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif backbone == "resnet34":
            self.model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        else:
            raise NotImplementedError("Invalid backbone.")
        fe_out_planes = self.model.fc.in_features
        self.model.avgpool = nn.AdaptiveMaxPool2d(1)
        self.model.fc = linear_layer(fe_out_planes, 2048, nn.ReLU())
        self.mu = linear_layer(2048, latent_dim)  # (latent_dim,)
        self.logvar = linear_layer(2048, latent_dim)  # (latent_dim,)

    def forward(self, batch):
        batch = self.model(batch)
        mu = self.mu(batch)
        logvar = self.logvar(batch)

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


class Sinusoid(nn.Linear):
    """Child class for sinusoidal encoding layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mu: float,
        sigma: float,
    ) -> None:
        """Initialize sinusoidal encoding layer.

        Args:
            in_features: number of input connections to the layer
            out_features: number of nodes in the layer
            mu: mean of normal distribution for weight initialization
            sigma: standard deviation of normal distribution for weight initialization
        """
        super().__init__(in_features, out_features, False)
        nn.init.normal_(self.weight, mu, sigma)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of sinusoidal encoding layer."""
        return torch.sin(nn.functional.linear(input, self.weight))


class CVAEPoREncoder(nn.Module):
    """Class for SE(3) to pose latent map."""

    def __init__(
        self, latent_dim: int, depth: int, breadth: int, conditional_encoder: bool
    ) -> None:
        """Initialize the pose encoder.

        Args:
            latent_dim: number of latent dimensions
            depth: number of hidden layers
            breadth: number of neurons in hidden layers
            conditional_encoder: if True, image features are taken as input
        """
        super().__init__()
        self._conditional_encoder = conditional_encoder
        self._latent_dim = latent_dim
        self.input_layer = linear_layer(
            9 + (512 if conditional_encoder else 0), breadth, nn.ReLU()
        )
        self.layers = nn.ModuleList(
            [linear_layer(breadth, breadth, nn.ReLU()) for _ in range(depth)]
        )
        self.mu = linear_layer(breadth, latent_dim)  # (latent_dim,)
        self.logvar = linear_layer(breadth, latent_dim)  # (latent_dim,)

    def forward(
        self, pose: torch.Tensor, img_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the encoder.

        Args:
            pose: translation vector and 6D rotation parameterization, shape (N, 9)
            img_feat: shape (N, 512)

        Returns:
            Mean of posterior Gaussian in latent space, shape (N, latent_dim)
            Logvariance of posterior Gaussian in latent space, shape (N, latent_dim)
        """
        batch = self.input_layer(
            torch.cat((pose, img_feat), dim=1) if self._conditional_encoder else pose
        )
        for layer in self.layers:
            batch = layer(batch)
        mu = self.mu(batch)
        logvar = self.logvar(batch)

        return mu, logvar

    def get_latent_dim(self) -> int:
        """Retrieve latent dimension of the model."""
        return self._latent_dim


class CVAEPoRDecoder(nn.Module):
    """Class for pose latent to SE(3) map conditioned on image."""

    def __init__(
        self,
        latent_dim: int,
        depth: int,
        breadth: int,
        bottleneck_dim: Optional[int] = None,
    ) -> None:
        """Initialize the conditional decoder.

        Args:
            latent_dim: number of latent dimensions
            depth: number of hidden layers
            breadth: number of neurons in hidden layers
            bottleneck: bottleneck dimension
        """
        super().__init__()
        self._latent_dim = latent_dim
        self._bottleneck_dim = bottleneck_dim
        if bottleneck_dim is None:
            self.input_layer = linear_layer(512 + latent_dim, breadth, nn.ReLU())
        else:
            self.img_input = linear_layer(512, bottleneck_dim, nn.ReLU())
            self.lat_input = linear_layer(latent_dim, bottleneck_dim, nn.ReLU())
            self.input_layer = linear_layer(bottleneck_dim, breadth, nn.ReLU())
        self.layers = nn.ModuleList(
            [linear_layer(breadth, breadth, nn.ReLU()) for _ in range(depth)]
        )
        self.fc_tvec = linear_layer(breadth, 3, nn.Sigmoid())
        self.fc_rvec = linear_layer(breadth, 6)

    def forward(self, lat_sample: torch.Tensor, img_feat: torch.Tensor):
        """Forward pass of the conditional decoder.

        Args:
            lat_sample: shape (N, M, L)
            img_feat: shape (N, 512)

        Returns:
            Predicted translation vector, shape (N*M, 3),
            Predicted 6D continuous rotation representation, shape (N*M, 6)
        """
        num_images, num_samples, _ = lat_sample.shape
        if self._bottleneck_dim is None:
            cat_feat = torch.cat(
                (img_feat[:, None, :].expand(-1, num_samples, -1), lat_sample),
                dim=2,
            )  # (N, M, 512 + L)
            batch = self.input_layer(cat_feat.flatten(end_dim=1))  # (N*M, breadth)
        else:
            img_feat = self.img_input(img_feat)  # (N, bottleneck_dim)
            lat_feat = self.lat_input(
                lat_sample.flatten(end_dim=1)
            )  # (N*M, bottleneck_dim)
            feat = lat_feat + img_feat[:, None, :].expand(-1, num_samples, -1).flatten(
                end_dim=1
            )  # (N*M, bottleneck_dim)
            batch = self.input_layer(feat)  # (N*M, breadth)

        for layer in self.layers:
            batch = layer(batch)  # (N*M, breadth)
        tvec = self.fc_tvec(batch)  # (N*M, 3)
        rvec = self.fc_rvec(batch)  # (N*M, 6)

        # return tvec.view(num_images, num_samples, 3), rvec.view(
        #     num_images, num_samples, 6
        # )
        return tvec, rvec

    def get_latent_dim(self) -> int:
        """Retrieve latent dimension of the model."""
        return self._latent_dim


class CVAEPoRBackbone(nn.Module):
    """Class for ResNet backbone."""

    def __init__(self, backbone="resnet18") -> None:
        """Initialize the conditional decoder.

        Args:
            backbone: Which ResNet size
        """
        super().__init__()

        # resnet bacbone to get image features
        if backbone == "resnet18":
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.model.avgpool = nn.AdaptiveAvgPool2d(1)
            self.model.fc = torch.nn.Identity()
        # elif backbone == "resnet34":
        #     self.model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        elif backbone == "vit_b_16":
            self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
            self.model.heads = torch.nn.Identity()
        else:
            raise NotImplementedError("Invalid backbone.")
        # fe_out_planes = self.model.fc.in_features
        # self.model.fc = linear_layer(fe_out_planes, 512, nn.ReLU())

    def forward(self, image: torch.Tensor):
        """Forward pass of the image bacbone.

        Args:
            image: shape (N, 3, H, W)

        Returns:
            image features, shape (N, 512)
        """
        return self.model(image)


class SimpleFlowModel(nn.Module):
    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, z):
        ld = 0.0
        for flow in self.flows:
            z, ld_ = flow(z)
            ld += ld_
        return z, ld


class CVAEPoRFlows(nn.Module):
    """Class for normalizing flows."""

    def __init__(self, latent_dim: int, num_flows: int, flow_type: str) -> None:
        """Initialize the flows module.

        Args:
            latent_dim: number of latent dimensions
            num_flows: Number of flow layers
            flow_type: Type of flow layers, either "planar" or "radial"
        """
        super().__init__()
        if flow_type == "planar":
            flow_layer = Planar
        elif flow_type == "radial":
            flow_layer = Radial
        else:
            raise ValueError("Unsupported flow type")

        self._flows = SimpleFlowModel(
            [flow_layer((latent_dim,), act="leaky_relu") for _ in range(num_flows)]
        )

    def forward(self, lat_sample: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of normalizing flows.

        Args:
            lat_sample: Initial samples, shape (N, L)

        Returns:
            Transformed samples, shape (N, L),
            log-determinant, shape (
        """
        return self._flows(lat_sample)
