"""
models.py -- Neural network architectures for scale-invariant optimization experiments.

Provides:
- CustomNormNoAffine: exact batch normalization without learnable parameters
- SimpleNormMLP: two-layer MLP with configurable normalization
- SmallConvBNNet: 3-block BN ConvNet for CIFAR experiments
- CifarResNetBN: CIFAR-style ResNet with affine-free BN for transfer experiments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomNormNoAffine(nn.Module):
    """Exact batch normalization without learnable affine parameters.

    Computes: (x - mean(x)) / std(x) with unbiased=False variance
    and a small numerical stabilizer eps.
    """

    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=0, keepdim=True)
        var = x.var(dim=0, unbiased=False, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps)


class SimpleNormMLP(nn.Module):
    """Two-layer MLP with normalization for scale-invariance experiments.

    Architecture: Linear -> Norm -> Softplus -> Linear
    Both linear layers have no bias. The first layer's rows are
    scale-invariant blocks when followed by normalization.

    Args:
        input_dim: input feature dimension (784 for MNIST)
        hidden_dim: hidden layer width
        num_classes: number of output classes
        norm_kind: "custom" for exact norm or "bn" for BatchNorm1d
        eps: numerical stabilizer for normalization
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 128,
                 num_classes: int = 10, norm_kind: str = "bn",
                 eps: float = 1e-5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        if norm_kind == "custom":
            self.norm = CustomNormNoAffine(eps=eps)
        else:
            self.norm = nn.BatchNorm1d(
                hidden_dim, affine=False, track_running_stats=False, eps=eps
            )
        self.act = nn.Softplus()
        self.fc2 = nn.Linear(hidden_dim, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.norm(h)
        h = self.act(h)
        return self.fc2(h)


class SmallConvBNNet(nn.Module):
    """Small 3-block BN ConvNet for CIFAR experiments.

    Architecture:
        Conv(3, c1, 3) -> BN -> Act -> AvgPool(2)
        Conv(c1, c2, 3) -> BN -> Act -> AvgPool(2)
        Conv(c2, c3, 3) -> BN -> Act -> GlobalAvgPool
        Linear(c3, num_classes)

    All convolutions are bias-free; all BN layers are affine-free with
    per-batch statistics. Each output-channel filter is treated as one
    scale-invariant block.

    Args:
        num_classes: output dimension (10 for CIFAR-10)
        channels: tuple of (c1, c2, c3) channel widths
        activation: "relu" or "softplus"
        eps: BN epsilon
    """

    def __init__(self, num_classes: int = 10,
                 channels: tuple = (32, 64, 128),
                 activation: str = "relu",
                 eps: float = 1e-5):
        super().__init__()
        c1, c2, c3 = channels

        self.conv1 = nn.Conv2d(3, c1, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1, affine=False,
                                   track_running_stats=False, eps=eps)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2, affine=False,
                                   track_running_stats=False, eps=eps)
        self.conv3 = nn.Conv2d(c2, c3, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(c3, affine=False,
                                   track_running_stats=False, eps=eps)
        self.classifier = nn.Linear(c3, num_classes, bias=False)

        act_fn = nn.ReLU if activation == "relu" else nn.Softplus
        self.act = act_fn()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.avg_pool2d(self.act(self.bn1(self.conv1(x))), 2)
        h = F.avg_pool2d(self.act(self.bn2(self.conv2(h))), 2)
        h = self.act(self.bn3(self.conv3(h)))
        h = F.adaptive_avg_pool2d(h, 1).flatten(1)
        return self.classifier(h)

    def conv_layers(self) -> dict:
        """Return dict mapping layer names to Conv2d modules."""
        return {
            'early conv1': self.conv1,
            'middle conv2': self.conv2,
            'late conv3': self.conv3,
        }


class BasicBlockNoAffine(nn.Module):
    """Basic residual block with affine-free batch normalization."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int,
                 stride: int = 1, eps: float = 1e-5):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(
            planes, affine=False, track_running_stats=False, eps=eps
        )
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(
            planes, affine=False, track_running_stats=False, eps=eps
        )

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(
                    planes, affine=False, track_running_stats=False, eps=eps
                ),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class CifarResNetBN(nn.Module):
    """CIFAR-style ResNet with affine-free BN and bias-free convolutions.

    This is a compact ResNet-18-style network adapted to 32x32 images:
    a 3x3 stem with no initial max-pool, followed by four residual stages.
    We track a small set of representative convolutions across depth rather
    than every convolutional kernel in the network.
    """

    def __init__(self, num_classes: int = 100,
                 widths: tuple = (64, 128, 256, 512),
                 blocks_per_stage: tuple = (2, 2, 2, 2),
                 eps: float = 1e-5):
        super().__init__()
        if len(widths) != 4 or len(blocks_per_stage) != 4:
            raise ValueError("widths and blocks_per_stage must both have length 4")

        self.in_planes = widths[0]
        self.stem = nn.Conv2d(
            3, widths[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.stem_bn = nn.BatchNorm2d(
            widths[0], affine=False, track_running_stats=False, eps=eps
        )
        self.layer1 = self._make_layer(widths[0], blocks_per_stage[0], stride=1, eps=eps)
        self.layer2 = self._make_layer(widths[1], blocks_per_stage[1], stride=2, eps=eps)
        self.layer3 = self._make_layer(widths[2], blocks_per_stage[2], stride=2, eps=eps)
        self.layer4 = self._make_layer(widths[3], blocks_per_stage[3], stride=2, eps=eps)
        self.classifier = nn.Linear(widths[3], num_classes, bias=False)

    def _make_layer(self, planes: int, num_blocks: int,
                    stride: int, eps: float) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for block_stride in strides:
            blocks.append(
                BasicBlockNoAffine(
                    self.in_planes, planes, stride=block_stride, eps=eps
                )
            )
            self.in_planes = planes
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.stem_bn(self.stem(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.classifier(out)

    def conv_layers(self) -> dict:
        """Return a representative depth-wise slice of tracked convolutions."""
        return {
            'stem conv': self.stem,
            'stage1 block1 conv1': self.layer1[0].conv1,
            'stage2 block1 conv1': self.layer2[0].conv1,
            'stage3 block1 conv1': self.layer3[0].conv1,
            'stage4 block1 conv1': self.layer4[0].conv1,
        }
