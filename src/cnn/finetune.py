"""Fine-tune ResNet18 on Flavia: unfreeze the last residual block and the new head.

The linear probe asks whether ImageNet features already separate the species; this asks
how much is gained by letting the network adapt them. Only `layer4` and the new 32-way
head are trained -- on 1430 training images, unfreezing the whole network would overfit,
and on a CPU it would also be far too slow. The early layers (edges, colour blobs,
textures) transfer as they are.

Two learning rates: the head starts from random weights and needs a large step, while
layer4 starts from a good solution and only needs nudging. One rate for both either
destroys the pretrained block or starves the head.

Usage:
    python -m src.cnn.finetune --epochs 6
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from . import backbones
from .data import LeafDataset, eval_transform, load_index, split, train_transform


def build_model(n_classes: int) -> nn.Module:
    from torchvision import models

    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for p in net.parameters():
        p.requires_grad = False
    for p in net.layer4.parameters():
        p.requires_grad = True
    net.fc = nn.Linear(net.fc.in_features, n_classes)
    return net


def run_epoch(net, loader, criterion, optimiser=None, device="cpu") -> tuple[float, float]:
    train = optimiser is not None
    net.train() if train else net.eval()
    total_loss, correct, seen = 0.0, 0, 0
    for i, (images, labels) in enumerate(loader, 1):
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(train):
            logits = net(images)
            loss = criterion(logits, labels)
        if train:
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        total_loss += loss.detach().item() * len(labels)
        correct += int((logits.argmax(1) == labels).sum())
        seen += len(labels)
        print(f"    {'train' if train else 'val  '} {i}/{len(loader)}"
              f"  loss {total_loss / seen:.3f}  acc {correct / seen:.3f}", end="\r", flush=True)
    print()
    return total_loss / seen, correct / seen


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune ResNet18 trên Flavia")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--checkpoint", type=Path, default=Path("cache/cnn/finetuned_resnet18.pt"))
    args = ap.parse_args()

    paths, labels, classes = load_index()
    train_idx, test_idx = split(labels)
    train_set = Subset(LeafDataset(paths, labels, train_transform()), train_idx)
    test_set = Subset(LeafDataset(paths, labels, eval_transform()), test_idx)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers)
    print(f"huấn luyện {len(train_idx)} ảnh, kiểm tra {len(test_idx)} ảnh, {len(classes)} lớp")

    net = build_model(len(classes))
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total = sum(p.numel() for p in net.parameters())
    print(f"tham số huấn luyện: {trainable/1e6:.1f}M / {total/1e6:.1f}M "
          f"({100*trainable/total:.0f}%)")

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW([
        {"params": net.layer4.parameters(), "lr": args.lr_backbone},
        {"params": net.fc.parameters(), "lr": args.lr_head},
    ], weight_decay=1e-4)

    history, best = [], 0.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        print(f"  epoch {epoch}/{args.epochs}")
        train_loss, train_acc = run_epoch(net, train_loader, criterion, optimiser)
        val_loss, val_acc = run_epoch(net, test_loader, criterion)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc,
                        "seconds": time.time() - start})
        print(f"    -> val acc {val_acc:.4f}  ({time.time() - start:.0f}s)")
        if val_acc > best:
            best = val_acc
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": net.state_dict(), "classes": classes,
                        "epoch": epoch, "val_acc": val_acc}, args.checkpoint)

    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(args.out / "cnn_finetune_history.csv", index=False)

    # Predictions of the best checkpoint, for the confusion matrix and the per-class
    # precision/recall the assignment asks for.
    net.load_state_dict(torch.load(args.checkpoint, weights_only=False)["state_dict"])
    net.eval()
    preds = []
    with torch.no_grad():
        for images, _ in test_loader:
            preds.append(net(images).argmax(1).cpu().numpy())
    np.savez_compressed(args.out / "cnn_finetune_predictions.npz",
                        y_true=labels[test_idx], y_pred=np.concatenate(preds),
                        classes=np.array(classes))
    print(f"\nval accuracy tốt nhất: {best:.4f}  -> {args.checkpoint}")


if __name__ == "__main__":
    main()
