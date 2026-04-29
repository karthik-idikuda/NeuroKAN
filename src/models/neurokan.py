import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

class KANLinear(nn.Module):
    """
    A simplified Kolmogorov-Arnold Layer.
    Instead of w*x + b, it learns a spline function phi(x) on the edge.
    Uses a Fourier expansion as a FastKAN proxy.
    """
    def __init__(self, in_features, out_features, grid_size=5):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.poly_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5))
        nn.init.uniform_(self.poly_weight, -0.02, 0.02)

    def forward(self, x):
        base_out = F.linear(x, self.base_weight)
        x_norm = torch.tanh(x)
        poly_out = torch.zeros_like(base_out)
        for i in range(self.grid_size):
            freq = np.pi * (i + 1)
            term = torch.sin(freq * x_norm)  # (batch, in_features)
            poly_out += F.linear(term, self.poly_weight[:, :, i])  # (batch, out_features)
        return base_out + poly_out


class NeuroKAN(nn.Module):
    """
    NeuroKAN: Hybrid EfficientNetV2-S + KAN Classifier Head.
    Replaces standard Dense layers with Kolmogorov-Arnold layers
    that learn non-linear spline functions on their edges.
    """
    def __init__(self, num_classes=4):
        super(NeuroKAN, self).__init__()

        self.backbone = models.efficientnet_v2_s(weights='DEFAULT')
        n_inputs = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.kan_head = nn.Sequential(
            nn.Dropout(p=0.3),
            KANLinear(n_inputs, 128, grid_size=5),
            nn.SiLU(),
            KANLinear(128, num_classes, grid_size=3)
        )

    def forward(self, x):
        features = self.backbone(x)
        out = self.kan_head(features)
        return out
