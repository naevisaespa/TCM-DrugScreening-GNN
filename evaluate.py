import argparse
from pathlib import Path

import torch

from data import build_pair_tensors, get_feature_and_metapath, load_graph, load_edge_df, validate_eval_feasibility
from model import Model
from utils import (
    append_metrics_csv,
    build_weighted_bce,
    compute_binary_metrics,
    compute_ranking_at_k,
    m2v,
    save_json,
    set_seed,
)


TRAIN_EDGE_SET = "hard_shared_1to10"
VALID_EDGE_SET = "hard_shared_1to10"
DEFAULT_TEST_EDGE_SETS = ("hard_shared_1to1", "random_1to10")
HIDDEN_FEATS = 64
NUM_HEADS = 4
DROPOUT = 0.6
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=1105)
    parser.add_argument("--hidden-feats", type=int, default=HIDDEN_FEATS)
    parser.add_argument("--num-heads", type=int, default=NUM_HEADS)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--test-edge-sets", default=",".join(DEFAULT_TEST_EDGE_SETS))
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


def forward_logits_batched(model, graph, feature, mdrug, mdis, drug_idx, disease_idx, batch_size=65536):
    total = int(drug_idx.shape[0])
    if total <= batch_size:
        return forward_logits(model, graph, feature, mdrug, mdis, drug_idx, disease_idx)

    logits_parts = []
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        logits_parts.append(
            forward_logits(
                model,
                graph,
                feature,
                mdrug,
                mdis,
                drug_idx[start:end],
                disease_idx[start:end],
            )
        )
    return torch.cat(logits_parts, dim=0)


def main():
    args = parse_args()
    set_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)

    print(f"graph_root={args.graph_root}")
    print(f"checkpoint={args.checkpoint}")
    print(f"test_edge_sets={args.test_edge_sets}")
    print("model=no_gat_gcn_mlp_m2v")

    graph_root = Path(args.graph_root)
    test_graph, _ = load_graph(graph_root / "test_graph.bin")
    test_graph = test_graph.to(device)
    test_feature, metapath = get_feature_and_metapath(test_graph)
    mdrug_test, mdis_test = m2v(test_graph, metapath, args.hidden_feats, device)

    model = build_model(
        test_graph,
        feature_dim=test_feature["drug"].shape[1],
        hidden_feats=args.hidden_feats,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    rows = []
    for test_edge_set in [x.strip() for x in args.test_edge_sets.split(",") if x.strip()]:
        test_df = load_edge_df(graph_root / f"test_edges_{test_edge_set}.csv", f"test:{test_edge_set}")
        validate_eval_feasibility(test_graph, test_df, f"test:{test_edge_set}")
        test_drug, test_disease, test_label = build_pair_tensors(test_df, device)
        criterion, _ = build_weighted_bce(test_label, device)

        with torch.no_grad():
            logits_test = forward_logits_batched(
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
            metrics = compute_binary_metrics(y_test, prob_test, criterion(logits_test, test_label).item())
            metrics.update(compute_ranking_at_k(test_df, prob_test))

        row = {
            "train_edge_set": TRAIN_EDGE_SET,
            "valid_edge_set": VALID_EDGE_SET,
            "test_edge_set": test_edge_set,
            "checkpoint": args.checkpoint,
            **metrics,
        }
        append_metrics_csv(save_dir / "dual_test_metrics.csv", row)
        rows.append(row)

    save_json(save_dir / "dual_test_metrics.json", {"rows": rows})
    print({"rows": rows})


if __name__ == "__main__":
    main()
