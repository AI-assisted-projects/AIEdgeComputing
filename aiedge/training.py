"""Training and evaluation.

One ``train`` function covers every case in the course:
  * ordinary training                   -> ``train(model, data)``
  * knowledge distillation              -> ``train(student, data, teacher=teacher)``
  * fine-tuning after edge pruning      -> ``train(model, data, masks=masks)``
  * quantization-aware training         -> ``train(qat_model, data)`` (the model
                                           itself contains the fake quantization)
"""

from __future__ import annotations

import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C


def set_seed(seed: int = 0) -> None:
    """Make results repeatable: same seed -> same random numbers -> same model."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def distillation_loss(student_logits, teacher_logits, labels, temperature=4.0, alpha=0.5):
    """Hinton et al. (2015):  alpha * CE(student, labels) + (1-alpha) * T^2 * KL(teacher_T || student_T)."""
    hard = F.cross_entropy(student_logits, labels)
    soft = F.kl_div(F.log_softmax(student_logits / temperature, dim=1),
                    F.softmax(teacher_logits / temperature, dim=1),
                    reduction="batchmean") * temperature**2
    return alpha * hard + (1 - alpha) * soft


def random_time_shift(x: torch.Tensor, max_shift: int = 6) -> torch.Tensor:
    """Data augmentation: move every gesture in the batch a few samples earlier/later.

    The trigger on the board never fires at exactly the same moment, so this
    teaches the network that *when* inside the window a gesture happens does
    not matter much. Edge values are repeated to fill the gap.
    """
    b = x.shape[0]
    w = x.view(b, C.WINDOW_SAMPLES, C.N_CHANNELS)
    shifts = torch.randint(-max_shift, max_shift + 1, (b,))
    t = torch.arange(C.WINDOW_SAMPLES)
    src = (t[None, :] - shifts[:, None]).clamp(0, C.WINDOW_SAMPLES - 1)   # (b, 50)
    out = torch.gather(w, 1, src[:, :, None].expand(-1, -1, C.N_CHANNELS))
    return out.reshape(b, -1)


def apply_masks(model: nn.Module, masks: dict[str, torch.Tensor] | None) -> None:
    """Force pruned weights back to exactly zero (used after every optimizer step)."""
    if not masks:
        return
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in masks:
                p.mul_(masks[name])


@torch.no_grad()
def predict_logits(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    model.eval()
    return model(X)


@torch.no_grad()
def accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    return float((predict_logits(model, X).argmax(1) == y).float().mean())


def confusion_matrix(y_true, y_pred, n_classes: int = C.N_CLASSES) -> np.ndarray:
    """``cm[i, j]`` = how many samples of true class i were predicted as class j."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(np.asarray(y_true), np.asarray(y_pred)):
        cm[t, p] += 1
    return cm


def train(model: nn.Module, data, epochs: int = 150, lr: float = 3e-3, batch_size: int = 64,
          weight_decay: float = 1e-4, patience: int = 25, teacher: nn.Module | None = None,
          temperature: float = 4.0, alpha: float = 0.5, masks=None, augment: bool = False,
          seed: int | None = 0, verbose: bool = True) -> dict:
    """Mini-batch training with the Adam optimizer and early stopping.

    After every epoch we measure accuracy on the *validation* set and remember
    the best weights. If validation accuracy has not improved for ``patience``
    epochs we stop, and the model is restored to its best epoch.

    Returns a history dict with per-epoch ``train_loss``, ``train_acc`` and ``val_acc``.
    """
    if seed is not None:
        set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if teacher is not None:
        teacher.eval()
    apply_masks(model, masks)

    history = {"train_loss": [], "train_acc": [], "val_acc": []}
    best_acc, best_state, bad_epochs = -1.0, None, 0
    n = len(data.X_train)
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(n)
        total_loss, correct = 0.0, 0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            xb, yb = data.X_train[idx], data.y_train[idx]
            if augment:
                xb = random_time_shift(xb)
            logits = model(xb)
            if teacher is None:
                loss = F.cross_entropy(logits, yb)
            else:
                with torch.no_grad():
                    teacher_logits = teacher(xb)   # the teacher sees the same (augmented) batch
                loss = distillation_loss(logits, teacher_logits, yb, temperature, alpha)
            opt.zero_grad()
            loss.backward()
            opt.step()
            apply_masks(model, masks)
            total_loss += loss.item() * len(idx)
            correct += int((logits.argmax(1) == yb).sum())

        val_acc = accuracy(model, data.X_val, data.y_val)
        history["train_loss"].append(total_loss / n)
        history["train_acc"].append(correct / n)
        history["val_acc"].append(val_acc)
        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"epoch {epoch:3d}  loss {total_loss / n:.4f}  "
                  f"train acc {correct / n:.3f}  val acc {val_acc:.3f}")

        if val_acc > best_acc:
            best_acc, best_state, bad_epochs = val_acc, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print(f"early stop at epoch {epoch}: no improvement for {patience} epochs")
                break

    model.load_state_dict(best_state)
    if verbose:
        print(f"best validation accuracy: {best_acc:.3f}")
    history["best_val_acc"] = best_acc
    return history
