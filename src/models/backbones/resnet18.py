"""ResNet18Adapted: torchvision ResNet-18, pretrained or from scratch."""

from __future__ import annotations

import torch.nn as nn

#: What a torchvision ImageNet model expects. Resizing to this keeps the
#: pretrained stem (7x7 stride-2 conv + stride-2 maxpool) operating at the scale
#: its filters were actually learned at.
IMAGENET_SIZE = 224


class ResNet18Adapted(nn.Module):
    """torchvision ResNet-18.

    Two ways to feed a 32x32 activation image to an ImageNet ResNet:

    - `pretrained=True`  -> UPSCALE THE INPUT to 224x224 and leave the network
      completely untouched. A stock ResNet-18 opens with a 7x7 stride-2 conv and
      a stride-2 maxpool; at 32x32 those would crush the image to 8x8 before the
      first residual block, so the usual fix is to replace conv1 and delete the
      maxpool. But that *throws away the pretrained stem* -- exactly the weights
      you loaded the checkpoint for. Resizing instead means every pretrained
      filter sees inputs at the scale it was trained on, and the model is used
      as-is, which is also what makes "pretrained" an honest label in the
      ablation table.

    - `pretrained=False` -> keep the network at native resolution and adapt the
      stem CIFAR-style (3x3 stride-1 conv, no maxpool). With random weights there
      is nothing to preserve, and upscaling 32x32 to 224x224 would be ~49x the
      compute for no information gain.

    The QKV field happens to carry exactly 3 channels (Q, K, V), so the
    pretrained conv1 kernel -- shaped (64, 3, 7, 7) -- is shape-compatible and
    needs no surgery.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        pretrained: bool = True,
        dropout: float = 0.0,
        in_ch: int = 3,
    ):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        if pretrained and in_ch != 3:
            # conv1's pretrained kernel is (64, 3, 7, 7). Any other channel count
            # would force us to discard or reshape it, which silently defeats the
            # point of loading pretrained weights -- fail loudly instead.
            raise ValueError(
                f"pretrained resnet18 needs exactly 3 input channels, got {in_ch}. "
                "Use pretrained_backbone=false."
            )

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=weights)

        self.resize_to = IMAGENET_SIZE if pretrained else None

        if not pretrained:
            # Random init: no pretrained stem to protect, so adapt it to the
            # native (small) image size rather than paying to upscale.
            net.conv1 = nn.Conv2d(in_ch, 64, 3, stride=1, padding=1, bias=False)
            net.maxpool = nn.Identity()

        net.fc = nn.Identity()
        self.net = net
        self.proj = nn.Linear(512, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.embed_dim = embed_dim

    def forward(self, x):                     # (N, 3, L, M)
        if self.resize_to is not None:
            # Bilinear, aligned corners off -- the standard torchvision resize.
            # `antialias=True` matters when DOWNsampling; we only ever upsample
            # here, but it is set for correctness if L or M ever exceeds 224.
            x = nn.functional.interpolate(
                x,
                size=(self.resize_to, self.resize_to),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        feats = self.net(x)                   # (N, 512)
        return self.proj(self.drop(feats))    # (N, E)
