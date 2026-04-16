"""SVTR-Tiny OCR model for line-level CTC recognition.

The model expects grayscale line images shaped as ``[batch, 1, 32, width]``.
It returns batch-first CTC log-probabilities shaped as
``[batch, time, num_classes]`` where ``time ~= width / 4``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


DEFAULT_ALPHABET = (
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
    " .,;:!?\"'()-[]/&"
)


@dataclass(frozen=True)
class SVTRTinyConfig:
    """Configuration matching the project technical specification."""

    alphabet: str = DEFAULT_ALPHABET
    blank_index: int = 0
    in_channels: int = 1
    img_height: int = 32
    max_width: int = 512
    embed_dim: int = 64
    depths: tuple[int, int, int] = (3, 3, 3)
    num_heads: tuple[int, int, int] = (2, 4, 8)
    dims: tuple[int, int, int] = (64, 128, 256)
    local_window_size: tuple[int, int] = (7, 11)
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attention_dropout: float = 0.0
    drop_path_rate: float = 0.05
    output_batch_first: bool = True

    @property
    def num_classes(self) -> int:
        return len(self.alphabet) + 1

    @property
    def max_sequence_width(self) -> int:
        return self.max_width // 4


class DropPath(nn.Module):
    """Stochastic depth per sample."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        random_tensor.div_(keep_prob)
        return x * random_tensor


class PatchEmbedding(nn.Module):
    """Two 3x3 stride-2 convolutions: [B, 1, 32, W] -> [B, 64, 8, W/4]."""

    def __init__(self, in_channels: int = 1, embed_dim: int = 64) -> None:
        super().__init__()
        mid_channels = embed_dim // 2
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=3,
                stride=(2, 2),
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(
                mid_channels,
                embed_dim,
                kernel_size=3,
                stride=(2, 2),
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class ConvMerging(nn.Module):
    """Height-only downsampling between SVTR stages."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=(2, 1),
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class Mlp(nn.Module):
    """Token-wise feed-forward network."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class WindowAttention2d(nn.Module):
    """Local window self-attention implemented with scaled_dot_product_attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: tuple[int, int],
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.attention_dropout = attention_dropout

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, x: Tensor) -> Tensor:
        bsz, height, width, channels = x.shape
        win_h, win_w = self.window_size
        pad_h = (win_h - height % win_h) % win_h
        pad_w = (win_w - width % win_w) % win_w

        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

        padded_h, padded_w = height + pad_h, width + pad_w
        num_win_h = padded_h // win_h
        num_win_w = padded_w // win_w

        windows = (
            x.view(bsz, num_win_h, win_h, num_win_w, win_w, channels)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(bsz * num_win_h * num_win_w, win_h * win_w, channels)
        )

        qkv = self.qkv(windows)
        qkv = qkv.view(
            qkv.shape[0],
            qkv.shape[1],
            3,
            self.num_heads,
            self.head_dim,
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)

        attn_dropout = self.attention_dropout if self.training else 0.0
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=attn_dropout,
        )
        attended = attended.transpose(1, 2).reshape(windows.shape[0], -1, channels)
        attended = self.proj(attended)
        attended = self.proj_drop(attended)

        x = (
            attended.view(bsz, num_win_h, num_win_w, win_h, win_w, channels)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(bsz, padded_h, padded_w, channels)
        )
        return x[:, :height, :width, :]


class GlobalAttention2d(nn.Module):
    """Global self-attention over all spatial tokens."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attention_dropout = attention_dropout

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, x: Tensor) -> Tensor:
        bsz, height, width, channels = x.shape
        tokens = x.reshape(bsz, height * width, channels)

        qkv = self.qkv(tokens)
        qkv = qkv.view(
            bsz,
            height * width,
            3,
            self.num_heads,
            self.head_dim,
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)

        attn_dropout = self.attention_dropout if self.training else 0.0
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=attn_dropout,
        )
        attended = attended.transpose(1, 2).reshape(bsz, height * width, channels)
        attended = self.proj(attended)
        attended = self.proj_drop(attended)
        return attended.reshape(bsz, height, width, channels)


class SVTRBlock(nn.Module):
    """SVTR transformer block with either local or global token mixing."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mixing: str,
        window_size: tuple[int, int] = (7, 11),
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        if mixing == "local":
            self.attn = WindowAttention2d(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                attention_dropout=attention_dropout,
                projection_dropout=dropout,
            )
        elif mixing == "global":
            self.attn = GlobalAttention2d(
                dim=dim,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                projection_dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown mixing mode: {mixing!r}")

        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim=dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x_bhwc = x.permute(0, 2, 3, 1).contiguous()
        x_bhwc = x_bhwc + self.drop_path(self.attn(self.norm1(x_bhwc)))
        x_bhwc = x_bhwc + self.drop_path(self.mlp(self.norm2(x_bhwc)))
        return x_bhwc.permute(0, 3, 1, 2).contiguous()


def _make_drop_path_rates(depths: Sequence[int], drop_path_rate: float) -> list[float]:
    total_blocks = sum(depths)
    if total_blocks <= 1:
        return [drop_path_rate]
    return torch.linspace(0.0, drop_path_rate, total_blocks).tolist()


class SVTRTinyOCR(nn.Module):
    """SVTR-Tiny OCR architecture for old printed book line recognition."""

    def __init__(
        self,
        config: SVTRTinyConfig | None = None,
        *,
        num_classes: int | None = None,
        alphabet: str | None = None,
    ) -> None:
        super().__init__()

        if config is None:
            config = SVTRTinyConfig(alphabet=alphabet or DEFAULT_ALPHABET)
        elif alphabet is not None:
            config = SVTRTinyConfig(**{**config.__dict__, "alphabet": alphabet})

        self.config = config
        self.alphabet = config.alphabet
        self.blank_index = config.blank_index
        self.num_classes = num_classes or config.num_classes
        self.output_batch_first = config.output_batch_first

        if config.img_height % 4 != 0:
            raise ValueError("img_height must be divisible by the patch stride of 4")

        self.patch_embed = PatchEmbedding(
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(
                1,
                config.embed_dim,
                config.img_height // 4,
                config.max_sequence_width,
            )
        )
        self.pos_drop = nn.Dropout(config.dropout)

        drop_rates = iter(_make_drop_path_rates(config.depths, config.drop_path_rate))

        self.stage1 = self._make_stage(
            dim=config.dims[0],
            depth=config.depths[0],
            num_heads=config.num_heads[0],
            mixing="local",
            window_size=config.local_window_size,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            drop_rates=drop_rates,
            mlp_ratio=config.mlp_ratio,
        )
        self.merge1 = ConvMerging(config.dims[0], config.dims[1])
        self.stage2 = self._make_stage(
            dim=config.dims[1],
            depth=config.depths[1],
            num_heads=config.num_heads[1],
            mixing="local",
            window_size=config.local_window_size,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            drop_rates=drop_rates,
            mlp_ratio=config.mlp_ratio,
        )
        self.merge2 = ConvMerging(config.dims[1], config.dims[2])
        self.stage3 = self._make_stage(
            dim=config.dims[2],
            depth=config.depths[2],
            num_heads=config.num_heads[2],
            mixing="global",
            window_size=config.local_window_size,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            drop_rates=drop_rates,
            mlp_ratio=config.mlp_ratio,
        )

        self.pool = nn.AdaptiveAvgPool2d((1, None))
        self.norm = nn.LayerNorm(config.dims[2])
        self.classifier = nn.Linear(config.dims[2], self.num_classes)

        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    @staticmethod
    def _make_stage(
        *,
        dim: int,
        depth: int,
        num_heads: int,
        mixing: str,
        window_size: tuple[int, int],
        dropout: float,
        attention_dropout: float,
        drop_rates: Iterable[float],
        mlp_ratio: float,
    ) -> nn.Sequential:
        blocks = [
            SVTRBlock(
                dim=dim,
                num_heads=num_heads,
                mixing=mixing,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
                drop_path=next(drop_rates),
            )
            for _ in range(depth)
        ]
        return nn.Sequential(*blocks)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _check_input(self, x: Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W] input, got shape {tuple(x.shape)}")
        if x.shape[1] != self.config.in_channels:
            raise ValueError(
                f"Expected {self.config.in_channels} input channel(s), got {x.shape[1]}"
            )
        if x.shape[2] != self.config.img_height:
            raise ValueError(
                f"Expected image height {self.config.img_height}, got {x.shape[2]}"
            )
        if x.shape[3] > self.config.max_width:
            raise ValueError(
                f"Input width {x.shape[3]} exceeds configured max_width={self.config.max_width}"
            )

    def forward_features(self, x: Tensor) -> Tensor:
        self._check_input(x)

        x = self.patch_embed(x)
        if x.shape[3] > self.pos_embed.shape[3]:
            raise ValueError(
                f"Patch sequence width {x.shape[3]} exceeds positional capacity "
                f"{self.pos_embed.shape[3]}"
            )

        x = x + self.pos_embed[:, :, :, : x.shape[3]]
        x = self.pos_drop(x)

        x = self.stage1(x)
        x = self.merge1(x)
        x = self.stage2(x)
        x = self.merge2(x)
        x = self.stage3(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.pool(x).squeeze(2)
        x = x.transpose(1, 2).contiguous()
        x = self.norm(x)
        logits = self.classifier(x)
        log_probs = F.log_softmax(logits, dim=-1)

        if self.output_batch_first:
            return log_probs
        return log_probs.transpose(0, 1).contiguous()

    def encode_text(self, text: str) -> list[int]:
        """Encode text for CTC targets. Blank is reserved at ``blank_index``."""

        if self.blank_index != 0:
            raise NotImplementedError("encode_text currently expects blank_index=0")

        char_to_idx = {char: idx + 1 for idx, char in enumerate(self.alphabet)}
        return [char_to_idx[char] for char in text if char in char_to_idx]

    def decode_indices(self, indices: Sequence[int], collapse_repeats: bool = True) -> str:
        """Greedy CTC index decoder for quick validation/debugging."""

        chars: list[str] = []
        prev_idx: int | None = None
        for idx in indices:
            if idx == self.blank_index:
                prev_idx = idx
                continue
            if collapse_repeats and idx == prev_idx:
                continue
            if idx < 1 or idx > len(self.alphabet):
                prev_idx = idx
                continue
            chars.append(self.alphabet[idx - 1])
            prev_idx = idx
        return "".join(chars)


def build_svtr_tiny_ocr(
    num_classes: int | None = None,
    alphabet: str | None = None,
    output_batch_first: bool = True,
) -> SVTRTinyOCR:
    """Factory used by training scripts and notebooks."""

    config = SVTRTinyConfig(
        alphabet=alphabet or DEFAULT_ALPHABET,
        output_batch_first=output_batch_first,
    )
    return SVTRTinyOCR(config=config, num_classes=num_classes)


if __name__ == "__main__":
    model = build_svtr_tiny_ocr()
    sample = torch.randn(2, 1, 32, 512)
    out = model(sample)
    print(f"Output shape: {tuple(out.shape)}")
