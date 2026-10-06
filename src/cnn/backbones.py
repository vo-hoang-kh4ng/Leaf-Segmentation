"""The three ImageNet-pretrained backbones compared in assignment 2.

Each entry gives the network, the dimensionality of the embedding it produces once the
classifier head is removed, and its parameter count -- the report compares accuracy
against model size, where these three span a factor of 53 (2.5M to 132.9M).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision import models


@dataclass
class Backbone:
    name: str
    model: nn.Module      # feature extractor: input (B,3,224,224) -> (B, dim)
    dim: int
    params: int           # parameters of the full pretrained network, before surgery


def _flatten(module: nn.Module) -> nn.Module:
    return nn.Sequential(module, nn.Flatten())


def build(name: str) -> Backbone:
    """Load a pretrained network and strip its ImageNet classifier.

    The head is removed rather than re-initialised: what we want from these networks is
    the representation, and the 1000-way ImageNet head is irrelevant to 32 leaf species.
    """
    if name == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        params = sum(p.numel() for p in net.parameters())
        dim = net.fc.in_features
        net.fc = nn.Identity()
        return Backbone(name, net, dim, params)

    if name == "mobilenet_v3_small":
        net = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        params = sum(p.numel() for p in net.parameters())
        dim = net.classifier[0].in_features
        net.classifier = nn.Identity()
        return Backbone(name, net, dim, params)

    if name == "vgg11_bn":
        net = models.vgg11_bn(weights=models.VGG11_BN_Weights.IMAGENET1K_V1)
        params = sum(p.numel() for p in net.parameters())
        # Keep the two 4096-unit fully connected layers, drop only the final classifier:
        # VGG's representation is conventionally taken from fc7, not from the conv stack.
        dim = net.classifier[6].in_features
        net.classifier[6] = nn.Identity()
        return Backbone(name, net, dim, params)

    raise ValueError(f"unknown backbone {name!r}")


NAMES = ("resnet18", "mobilenet_v3_small", "vgg11_bn")
PRETTY = {"resnet18": "ResNet18",
          "mobilenet_v3_small": "MobileNetV3-Small",
          "vgg11_bn": "VGG11-BN"}


@torch.no_grad()
def embed(backbone: Backbone, loader, device: str = "cpu") -> torch.Tensor:
    backbone.model.eval().to(device)
    out = []
    for i, (images, _labels) in enumerate(loader, 1):
        out.append(backbone.model(images.to(device)).cpu())
        print(f"    batch {i}/{len(loader)}", end="\r", flush=True)
    print()
    return torch.cat(out)
