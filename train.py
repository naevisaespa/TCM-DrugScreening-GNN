import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from data import build_pair_tensors, get_feature_and_metapath, load_split_bundle
from model import Model
from utils import (
    JointMetricEarlyStopping,
    append_metrics_csv,
    build_weighted_bce,
    compute_binary_metrics,
    compute_ranking_at_k,
    get_metrics_auc,
    m2v,
    save_json,
    set_seed,
)


TRAIN_EDGE_SET = "hard_shared_1to10"
VALID_EDGE_SET = "hard_shared_1to10"
TEST_EDGE_SET = "random_1to10"
HIDDEN_FEATS = 64
NUM_HEADS = 4
DROPOUT = 0.6
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001
MAX_EPOCHS = 400
PATIENCE = 80
GRAD_CLIP_MAX_NORM = 5.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-root", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=1105)
    parser.add_argument("--epoch", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--hidden-feats", type=int, default=HIDDEN_FEATS)
    parser.add_argument("--num-heads", type=int, default=NUM_HEADS)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--test-edge-set", default=TEST_EDGE_SET)
    return parser.parse_args()


def get_device(device_arg):
    if str(device_arg) in {"", "-1", "cpu"}:
        return torch.device("cpu")
    return torch.device(f"cuda:{device_arg}")


def build_model(graph, feature_dim, hidden_feats, num_heads, dropout):
    return Model(
        etypes=graph.etypes,
        ntypes=graph.ntypes,
        in_feats=feature_dim,
        hidden_feats=hidden_feats,
        num_heads=num_heads,
        dropout=dropout,
    )


def forward_logits(model, graph, feature, mdrug, mdis, drug_idx, disease_idx):
    logits, _, _ = model(
        graph,
        feature,
        mdrug=mdrug,
        mdis=mdis,
        pair_drug_idx=drug_idx,
        pair_disease_idx=disease_idx,
    )
    return logits.view(-1)


def plot_curves(save_dir, train_values, valid_values, name):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_values) + 1), train_values, label=f"Train {name}")
    plt.plot(range(1, len(valid_values) + 1), valid_values, label=f"Valid {name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(Path(save_dir) / f"{name.lower()}_curve.pdf")
    plt.close()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = get_device(args.device)

    print(f"graph_root={args.graph_root}")
    print("train_protocol=structure-based_1to10")
    print("final_model=heterogeneous_gcn_with_m2v_and_mlp")

    bundle = load_split_bundle(args.graph_root, TRAIN_EDGE_SET, VALID_EDGE_SET, args.test_edge_set)
    train_graph = bundle["train_graph"].to(device)
    valid_graph = bundle["valid_graph"].to(device)
    test_graph = bundle["test_graph"].to(device)
    train_df = bundle["train_df"]
    valid_df = bundle["valid_df"]
    test_df = bundle["test_df"]

    train_feature, metapath = get_feature_and_metapath(train_graph)
    valid_feature, _ = get_feature_and_metapath(valid_graph)
    test_feature, _ = get_feature_and_metapath(test_graph)

    mdrug_train, mdis_train = m2v(train_graph, metapath, args.hidden_feats, device)
    mdrug_valid, mdis_valid = m2v(valid_graph, metapath, args.hidden_feats, device)
    mdrug_test, mdis_test = m2v(test_graph, metapath, args.hidden_feats, device)

    train_drug, train_disease, train_label = build_pair_tensors(train_df, device)
    valid_drug, valid_disease, valid_label = build_pair_tensors(valid_df, device)
    test_drug, test_disease, test_label = build_pair_tensors(test_df, device)

    model = build_model(
        train_graph,
        feature_dim=train_feature["drug"].shape[1],
        hidden_feats=args.hidden_feats,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_criterion, train_pos_weight = build_weighted_bce(train_label, device)
    valid_criterion, valid_pos_weight = build_weighted_bce(valid_label, device)
    stopper = JointMetricEarlyStopping(args.patience, args.save_dir)

    train_loss_list, valid_loss_list = [], []
    train_auc_list, valid_auc_list = [], []
    train_aupr_list, valid_aupr_list = [], []

    for epoch in range(1, args.epoch + 1):
        model.train()
        logits_train = forward_logits(
            model,
            train_graph,
            train_feature,
            mdrug_train,
            mdis_train,
            train_drug,
            train_disease,
        )
        loss = train_criterion(logits_train, train_label)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
        optimizer.step()
        train_loss_list.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            prob_train = torch.sigmoid(logits_train).cpu().numpy()
            y_train = train_label.cpu().numpy()
            train_auc, train_aupr = get_metrics_auc(y_train, prob_train)
            train_auc_list.append(train_auc)
            train_aupr_list.append(train_aupr)

            logits_valid = forward_logits(
                model,
                valid_graph,
                valid_feature,
                mdrug_valid,
                mdis_valid,
                valid_drug,
                valid_disease,
            )
            prob_valid = torch.sigmoid(logits_valid).cpu().numpy()
            y_valid = valid_label.cpu().numpy()
            valid_auc, valid_aupr = get_metrics_auc(y_valid, prob_valid)
            train_auc_list[-1] = train_auc
            valid_auc_list.append(valid_auc)
            train_aupr_list[-1] = train_aupr
            valid_aupr_list.append(valid_aupr)
            valid_loss = valid_criterion(logits_valid, valid_label)
            valid_loss_list.append(float(valid_loss.item()))
            early_stop = stopper.step(float(valid_loss.item()), valid_auc, valid_aupr, model)

        if epoch % 20 == 0:
            print(
                f"Epoch {epoch} Loss: {loss.item():.3f}; Train AUC: {train_auc:.3f}; "
                f"Train AUPR: {train_aupr:.3f}; Valid AUC: {valid_auc:.3f}; Valid AUPR: {valid_aupr:.3f}"
            )
        if early_stop:
            break

    plot_curves(args.save_dir, train_loss_list, valid_loss_list, "Loss")
    plot_curves(args.save_dir, train_auc_list, valid_auc_list, "AUC")
    plot_curves(args.save_dir, train_aupr_list, valid_aupr_list, "AUPR")

    model.load_state_dict(torch.load(stopper.filename, map_location=device))
    model.eval()
    test_criterion, test_pos_weight = build_weighted_bce(test_label, device)
    with torch.no_grad():
        logits_test = forward_logits(
            model,
            test_graph,
            test_feature,
            mdrug_test,
            mdis_test,
            test_drug,
            test_disease,
        )
        prob_test = torch.sigmoid(logits_test).cpu().numpy()
        y_test = test_label.cpu().numpy()
        metrics = compute_binary_metrics(y_test, prob_test, test_criterion(logits_test, test_label).item())
        metrics.update(compute_ranking_at_k(test_df, prob_test))

    manifest = {
        "dataset": "CHIdataset",
        "graph_root": args.graph_root,
        "train_edge_set": TRAIN_EDGE_SET,
        "valid_edge_set": VALID_EDGE_SET,
        "test_edge_set": args.test_edge_set,
        "model": "no_gat_gcn_mlp_m2v",
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "hidden_feats": args.hidden_feats,
        "num_heads": args.num_heads,
        "optimizer": "Adam",
        "gradient_clip_max_norm": GRAD_CLIP_MAX_NORM,
        "train_pos_weight": train_pos_weight,
        "valid_pos_weight": valid_pos_weight,
        "test_pos_weight": test_pos_weight,
        "best_valid_loss": stopper.best_loss,
        "best_valid_auc": stopper.best_auc,
        "best_valid_aupr": stopper.best_aupr,
        "checkpoint": stopper.filename,
        "test_metrics": metrics,
    }
    save_json(Path(args.save_dir) / "train_manifest.json", manifest)
    append_metrics_csv(
        Path(args.save_dir) / "train_test_metrics.csv",
        {
            "train_edge_set": TRAIN_EDGE_SET,
            "valid_edge_set": VALID_EDGE_SET,
            "test_edge_set": args.test_edge_set,
            "checkpoint": stopper.filename,
            **metrics,
        },
    )
    print(metrics)


if __name__ == "__main__":
    main()
