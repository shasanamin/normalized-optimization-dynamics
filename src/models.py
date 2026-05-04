"""
models.py -- Neural network architectures for scale-invariant optimization experiments.

Provides:
- CustomNormNoAffine: exact batch normalization without learnable parameters
- SimpleNormMLP: two-layer MLP with configurable normalization
- SmallConvBNNet: 3-block BN ConvNet for CIFAR experiments
- CifarResNetBN: CIFAR-style ResNet with affine-free BN for transfer experiments
- GPT2NoAffine: GPT-2-style causal LM with affine-free LayerNorm
"""

import math
from typing import Optional

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


class CausalSelfAttentionNoAffine(nn.Module):
    """GPT-style causal self-attention with bias-free projections."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attn_pdrop: float = 0.0,
        resid_pdrop: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.c_proj = nn.Linear(d_model, d_model, bias=bias)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

    def forward(self, x: torch.Tensor):
        batch, seq_len, channels = x.shape

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
        ).view(1, 1, seq_len, seq_len)
        scores = scores.masked_fill(~causal_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        return self.resid_dropout(self.c_proj(y))


class GPT2MLPNoAffine(nn.Module):
    """Transformer feed-forward block with bias-free linear maps."""

    def __init__(
        self,
        d_model: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        hidden_dim = mlp_ratio * d_model
        self.c_fc = nn.Linear(d_model, hidden_dim, bias=bias)
        self.c_proj = nn.Linear(hidden_dim, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x, approximate="tanh")
        x = self.c_proj(x)
        return self.dropout(x)


class GPT2BlockNoAffine(nn.Module):
    """Pre-LayerNorm GPT block with affine-free LayerNorm."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: int = 4,
        eps: float = 1e-5,
        attn_pdrop: float = 0.0,
        resid_pdrop: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model, eps=eps, elementwise_affine=False)
        self.attn = CausalSelfAttentionNoAffine(
            d_model=d_model,
            n_heads=n_heads,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            bias=bias,
        )
        self.ln_2 = nn.LayerNorm(d_model, eps=eps, elementwise_affine=False)
        self.mlp = GPT2MLPNoAffine(
            d_model=d_model,
            mlp_ratio=mlp_ratio,
            dropout=resid_pdrop,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2NoAffine(nn.Module):
    """GPT-2-style causal LM used for LayerNorm validation experiments.

    The tracked scale-invariant blocks are the attention projections immediately
    downstream of affine-free LayerNorm: fused QKV ``c_attn`` and output
    projection ``c_proj`` in every transformer block.
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        n_layers: int = 12,
        n_heads: int = 12,
        d_model: int = 768,
        mlp_ratio: int = 4,
        eps: float = 1e-5,
        embd_pdrop: float = 0.0,
        attn_pdrop: float = 0.0,
        resid_pdrop: float = 0.0,
        tie_weights: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.d_model = d_model

        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(embd_pdrop)
        self.blocks = nn.ModuleList([
            GPT2BlockNoAffine(
                d_model=d_model,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                eps=eps,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                bias=bias,
            )
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model, eps=eps, elementwise_affine=False)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> dict:
        batch, seq_len = idx.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}"
            )

        pos = torch.arange(0, seq_len, device=idx.device, dtype=torch.long)
        pos = pos.unsqueeze(0).expand(batch, -1)
        x = self.wte(idx) + self.wpe(pos)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return {"logits": logits, "loss": loss}

    def tracked_layers(self) -> dict:
        """Return attention projections treated as scale-invariant blocks."""
        layers = {}
        for i, block in enumerate(self.blocks):
            layers[f"block{i}.attn.c_attn"] = block.attn.c_attn
            layers[f"block{i}.attn.c_proj"] = block.attn.c_proj
        return layers


def small_gpt2_noaffine(
    vocab_size: int,
    max_seq_len: int = 256,
    eps: float = 1e-5,
) -> GPT2NoAffine:
    """4-block, d_model=256 GPT-2-style model used for WikiText."""
    return GPT2NoAffine(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        n_layers=4,
        n_heads=4,
        d_model=256,
        mlp_ratio=4,
        eps=eps,
        tie_weights=True,
        bias=False,
    )


def gpt2_noaffine(
    vocab_size: int,
    max_seq_len: int = 256,
    eps: float = 1e-5,
) -> GPT2NoAffine:
    """12-block, d_model=768 GPT-2-style model used for OpenWebText."""
    return GPT2NoAffine(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        n_layers=12,
        n_heads=12,
        d_model=768,
        mlp_ratio=4,
        eps=eps,
        tie_weights=True,
        bias=False,
    )


# Backward-compatible aliases for the collaborator's original script names.
tiny_gpt2_noaffine = small_gpt2_noaffine
normal_gpt2_noaffine = gpt2_noaffine
