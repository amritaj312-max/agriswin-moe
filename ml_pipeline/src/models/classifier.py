"""
The one model structure every run shares:

    image -> timm backbone (pooled features) -> Dropout(p) -> Linear(num_classes)

The head is built here rather than by timm because timm places its dropout
inconsistently. Swin-V2, MobileViT and FastViT get an nn.Dropout module, but
EfficientNet applies F.dropout inline, tied to model.training. MC-dropout
(Novelty 3) works by switching Dropout modules back on at inference while
everything else -- BatchNorm included -- stays in eval mode, so on EfficientNet
it would silently do nothing and report zero uncertainty. Owning the head makes
it identical for every backbone, and it is where Phase 3's MoE head plugs in.

Stochastic depth (drop_path_rate) is likewise made to mean the same thing for
every backbone: randomly skipping a residual branch, as Huang et al. (2016)
define it. timm's byobnet models, MobileViT among them, also apply it inside
blocks that have no shortcut, where it zeroes the block's whole output -- the
image is wiped rather than a branch skipped. On MobileViT-S that hit about 1 in
5 training images per step and held training accuracy near 85%. Such layers
are switched off here; on EfficientNet, Swin-V2 and FastViT there are none, so
for them nothing changes.

A custom backbone only needs registering with timm (@register_model) to be
trainable through this class unchanged.
"""

import timm
import torch.nn as nn
from timm.layers import DropPath


def disable_residual_less_drop_path(model):
    """Replace drop-path in blocks without a shortcut by a no-op; returns how
    many were replaced. Drop-path has no weights, so checkpoints still load."""
    replaced = 0
    for block in model.modules():
        if isinstance(getattr(block, "drop_path", None), DropPath) and getattr(block, "shortcut", False) is None:
            block.drop_path = nn.Identity()
            replaced += 1
    return replaced


class Classifier(nn.Module):
    def __init__(self, backbone, num_classes, pretrained=True, drop_rate=0.2, drop_path_rate=0.1):
        super().__init__()
        # num_classes=0: timm returns pooled features and drops its own head
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0,
                                          drop_rate=0.0, drop_path_rate=drop_path_rate)
        disable_residual_less_drop_path(self.backbone)
        features = getattr(self.backbone, "head_hidden_size", None) or self.backbone.num_features
        self.dropout = nn.Dropout(drop_rate)
        self.head = nn.Linear(features, num_classes)

    def forward(self, x):
        return self.head(self.dropout(self.backbone(x)))

    def data_config(self):
        """Input normalisation and resolution the backbone was pretrained with."""
        return timm.data.resolve_model_data_config(self.backbone)
