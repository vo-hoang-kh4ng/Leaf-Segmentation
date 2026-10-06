"""What the network looks at: first-layer filters, feature maps, Grad-CAM, t-SNE.

The assignment asks for illustrations of the features, which for a CNN means showing the
learned representation rather than listing formulas. Four views, each answering a
different question:

  filters    what the first convolution responds to (edges, colour opponency)
  maps       which of those responses fire on one real leaf
  gradcam    which region drove the final decision -- the check that the network looks
             at the leaf and not at a scanner artefact
  tsne       whether the 512-d embedding separates the species before any classifier

Grad-CAM weights each channel of the last convolutional block by the gradient of the
predicted class with respect to that channel, then sums: regions whose activation would
most increase the score light up.

Usage:
    python -m src.cnn.visualise
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import MEAN, STD, eval_transform, load_index

DEMO_IMAGE = "data/flavia/1083.jpg"


def _denormalise(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.numpy().transpose(1, 2, 0) * np.array(STD) + np.array(MEAN)
    return np.clip(image, 0, 1)


def load_finetuned(checkpoint: Path, n_classes: int):
    from .finetune import build_model

    net = build_model(n_classes)
    if checkpoint.exists():
        state = torch.load(checkpoint, weights_only=False)
        net.load_state_dict(state["state_dict"])
        print(f"  dùng mô hình đã fine-tune (val acc {state['val_acc']:.4f})")
    else:
        print(f"  chưa có {checkpoint}, dùng trọng số ImageNet")
    return net.eval()


def filters_figure(net, out: Path) -> None:
    """The 64 first-layer kernels, 7x7x3, shown as RGB patches."""
    import matplotlib.pyplot as plt

    weights = net.conv1.weight.detach().clone()
    weights = (weights - weights.min()) / (weights.max() - weights.min())
    fig, axes = plt.subplots(8, 8, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(weights[i].numpy().transpose(1, 2, 0))
        ax.axis("off")
    fig.suptitle("64 bộ lọc tầng tích chập đầu tiên (ResNet18)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def maps_figure(net, image: torch.Tensor, out: Path) -> None:
    """First-layer responses on one leaf: the 12 strongest channels."""
    import matplotlib.pyplot as plt

    with torch.no_grad():
        maps = net.conv1(image.unsqueeze(0))[0]
    strength = maps.flatten(1).abs().mean(1)
    top = torch.argsort(strength, descending=True)[:12]

    fig, axes = plt.subplots(2, 7, figsize=(13, 4))
    axes[0, 0].imshow(_denormalise(image))
    axes[0, 0].set_title("ảnh đầu vào", fontsize=9)
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")
    for ax, channel in zip(list(axes[0, 1:]) + list(axes[1, 1:]), top):
        ax.imshow(maps[channel].numpy(), cmap="viridis")
        ax.set_title(f"kênh {int(channel)}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Bản đồ đặc trưng tầng 1 — 12 kênh phản ứng mạnh nhất", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def gradcam_figure(net, image: torch.Tensor, classes: list[str], out: Path) -> None:
    import matplotlib.pyplot as plt

    activations, gradients = {}, {}

    # The hooks must return None: whatever a forward hook returns replaces the layer's
    # output, and whatever a backward hook returns replaces grad_input -- returning the
    # captured tensor silently corrupts the pass (and here crashed it outright).
    def save_activation(module, inputs, output) -> None:
        activations["a"] = output

    def save_gradient(module, grad_input, grad_output) -> None:
        gradients["g"] = grad_output[0]

    target = net.layer4[-1]
    h1 = target.register_forward_hook(save_activation)
    h2 = target.register_full_backward_hook(save_gradient)

    logits = net(image.unsqueeze(0))
    predicted = int(logits.argmax(1))
    net.zero_grad()
    logits[0, predicted].backward()
    h1.remove()
    h2.remove()

    weights = gradients["g"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * activations["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=image.shape[1:], mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    axes[0].imshow(_denormalise(image))
    axes[0].set_title("ảnh đầu vào", fontsize=9)
    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("bản đồ Grad-CAM", fontsize=9)
    axes[2].imshow(_denormalise(image))
    axes[2].imshow(cam, cmap="jet", alpha=0.45)
    axes[2].set_title(f"chồng lên nhau — dự đoán: {classes[predicted]}", fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Grad-CAM: vùng ảnh quyết định dự đoán", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def tsne_figure(cache: Path, out: Path) -> None:
    """t-SNE of the frozen ResNet18 embedding, coloured by species."""
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    data = np.load(cache / "resnet18.npz", allow_pickle=False)
    X, y = data["X"], data["y"]
    Z = TSNE(n_components=2, perplexity=30, init="pca", random_state=42).fit_transform(X)

    palette = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.scatter(Z[:, 0], Z[:, 1], c=[palette[i % len(palette)] for i in y], s=6,
               alpha=0.8, linewidths=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("t-SNE của embedding ResNet18 (đóng băng) — mỗi màu là một loài")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Hình minh hoạ đặc trưng của CNN")
    ap.add_argument("--image", type=Path, default=Path(DEMO_IMAGE))
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("cache/cnn/finetuned_resnet18.pt"))
    ap.add_argument("--cache", type=Path, default=Path("cache/cnn"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")

    _paths, labels, classes = load_index()
    net = load_finetuned(args.checkpoint, len(classes))

    from PIL import Image
    image = eval_transform()(Image.open(args.image).convert("RGB"))

    args.out.mkdir(parents=True, exist_ok=True)
    filters_figure(net, args.out / "cnn_filters.png")
    maps_figure(net, image, args.out / "cnn_feature_maps.png")
    gradcam_figure(net, image, classes, args.out / "cnn_gradcam.png")
    tsne_figure(args.cache, args.out / "cnn_tsne.png")
    print("-> results/cnn_{filters,feature_maps,gradcam,tsne}.png")


if __name__ == "__main__":
    main()
