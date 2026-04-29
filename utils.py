import csv
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from dgl.nn.pytorch import MetaPath2Vec
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim import SparseAdam
from torch.utils.data import DataLoader


TOPK_VALUES = (50, 100, 200)
METRIC_FIELDNAMES = [
    "train_edge_set",
    "valid_edge_set",
    "test_edge_set",
    "checkpoint",
    "loss",
    "auc",
    "aupr",
    "acc",
    "f1",
    "precision",
    "recall",
    "specificity",
    "mcc",
    "precision_at_50",
    "recall_at_50",
    "precision_at_100",
    "recall_at_100",
    "precision_at_200",
    "recall_at_200",
]


class JointMetricEarlyStopping:
    def __init__(self, patience, save_dir):
        dt = datetime.now()
        self.filename = str(
            Path(save_dir) / f"early_stop_metric_{dt.date()}_{dt.hour:02d}-{dt.minute:02d}-{dt.second:02d}.pth"
        )
        self.patience = patience
        self.counter = 0
        self.best_aupr = None
        self.best_auc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, auc, aupr, model):
        improved = False
        if self.best_aupr is None:
            improved = True
        elif aupr > self.best_aupr + 1e-6:
            improved = True
        elif abs(aupr - self.best_aupr) <= 1e-6 and auc > self.best_auc + 1e-6:
            improved = True
        elif abs(aupr - self.best_aupr) <= 1e-6 and abs(auc - self.best_auc) <= 1e-6 and loss < self.best_loss - 1e-6:
            improved = True

        if improved:
            self.best_aupr = aupr
            self.best_auc = auc
            self.best_loss = loss
            torch.save(model.state_dict(), self.filename)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_metrics_auc(y_true, prob):
    return float(roc_auc_score(y_true, prob)), float(average_precision_score(y_true, prob))


def build_weighted_bce(labels, device):
    num_neg = int((labels == 0).sum().item())
    num_pos = int((labels == 1).sum().item())
    if num_pos == 0:
        raise ValueError("Positive label count is 0; cannot build BCEWithLogitsLoss.")
    pos_weight = torch.tensor(float(num_neg / num_pos), dtype=torch.float32, device=device)
    return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight), float(pos_weight.item())


def m2v(graph, metapath, hidden_feats, device):
    model = MetaPath2Vec(graph, metapath, emb_dim=hidden_feats, window_size=3).to(device)
    dataloader = DataLoader(
        torch.arange(graph.num_nodes("disease")),
        batch_size=128,
        shuffle=True,
        collate_fn=model.sample,
    )
    optimizer = SparseAdam(model.parameters(), lr=0.025)

    for _ in range(1):
        for pos_u, pos_v, neg_v in dataloader:
            pos_u = pos_u.to(device)
            pos_v = pos_v.to(device)
            neg_v = neg_v.to(device)
            loss = model(pos_u, pos_v, neg_v)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    drug_nids = torch.LongTensor(model.local_to_global_nid["drug"]).to(device)
    disease_nids = torch.LongTensor(model.local_to_global_nid["disease"]).to(device)
    drug_emb = model.node_embed(drug_nids).detach()
    disease_emb = model.node_embed(disease_nids).detach()
    return drug_emb, disease_emb


def compute_binary_metrics(y_true, prob, loss):
    pred_labels = (prob > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred_labels).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "loss": float(loss),
        "auc": float(roc_auc_score(y_true, prob)),
        "aupr": float(average_precision_score(y_true, prob)),
        "acc": float(accuracy_score(y_true, pred_labels)),
        "f1": float(f1_score(y_true, pred_labels)),
        "precision": float(precision_score(y_true, pred_labels)),
        "recall": float(recall_score(y_true, pred_labels)),
        "specificity": float(specificity),
        "mcc": float(matthews_corrcoef(y_true, pred_labels)),
    }


def compute_ranking_at_k(edge_df, prob, ks=TOPK_VALUES):
    df = edge_df[["drug_idx", "disease_idx", "label"]].copy()
    df["score"] = prob
    rows = {}
    for k in ks:
        precision_values = []
        recall_values = []
        for _, group in df.groupby("disease_idx", sort=False):
            group = group.sort_values("score", ascending=False)
            topk = group.head(k)
            hits = float(topk["label"].sum())
            pos_total = float(group["label"].sum())
            precision_values.append(hits / float(k))
            recall_values.append(hits / pos_total if pos_total > 0 else 0.0)
        rows[f"precision_at_{k}"] = float(sum(precision_values) / len(precision_values))
        rows[f"recall_at_{k}"] = float(sum(recall_values) / len(recall_values))
    return rows


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def append_metrics_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
