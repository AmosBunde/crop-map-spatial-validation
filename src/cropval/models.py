"""Two models on the same per-field time series.

rf    Random forest on the flattened series: the strong, cheap baseline.
tcn   A small temporal convolutional network in PyTorch that reads the
      series as (features x dates). Small enough to train on a laptop CPU.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from torch import nn


def fit_predict_rf(x_train, y_train, x_test, n_classes: int, seed: int = 0) -> np.ndarray:
    rf = RandomForestClassifier(
        n_estimators=500, class_weight="balanced_subsample", n_jobs=-1, random_state=seed
    )
    rf.fit(x_train.reshape(len(x_train), -1), y_train)
    proba = np.zeros((len(x_test), n_classes), dtype=np.float32)
    proba[:, rf.classes_] = rf.predict_proba(x_test.reshape(len(x_test), -1))
    return proba


class TemporalCNN(nn.Module):
    def __init__(self, n_features: int, n_classes: int, width: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, width, 3, padding=1), nn.BatchNorm1d(width), nn.ReLU(),
            nn.Conv1d(width, width, 3, padding=1), nn.BatchNorm1d(width), nn.ReLU(),
            nn.Conv1d(width, width, 3, padding=1), nn.BatchNorm1d(width), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(width, n_classes),
        )

    def forward(self, x):  # x: (batch, dates, features)
        return self.net(x.transpose(1, 2))


def fit_predict_tcn(x_train, y_train, x_test, n_classes: int, seed: int = 0, epochs: int = 60) -> np.ndarray:
    torch.manual_seed(seed)
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True) + 1e-6
    xt = torch.tensor((x_train - mean) / std, dtype=torch.float32)
    xs = torch.tensor((x_test - mean) / std, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)

    counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    weight = torch.tensor(np.where(counts > 0, counts.sum() / (n_classes * np.maximum(counts, 1)), 0.0))
    model = TemporalCNN(x_train.shape[2], n_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * max(1, -(-len(xt) // 128)))
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        for idx in torch.randperm(len(xt), generator=gen).split(128):
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss_fn(model(xt[idx]), yt[idx]).backward()
            opt.step()
            sched.step()
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(xs), dim=1).numpy()


MODELS = {"rf": fit_predict_rf, "tcn": fit_predict_tcn}
