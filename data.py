from pathlib import Path

import dgl
import pandas as pd
import torch


def load_graph(graph_path):
    graphs, labels = dgl.load_graphs(str(graph_path))
    if not graphs:
        raise RuntimeError(f"No graph found in {graph_path}")
    return graphs[0], labels


def load_edge_df(csv_path, split_name):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {split_name} edges: {csv_path}")
    df = pd.read_csv(csv_path, dtype={"drug_idx": int, "disease_idx": int, "label": int})
    required_cols = {"drug_idx", "disease_idx", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{split_name} edges must contain columns: {sorted(required_cols)}")
    return df


def validate_eval_feasibility(graph, edge_df, split_name):
    if edge_df["drug_idx"].max() >= graph.num_nodes("drug"):
        raise RuntimeError(f"{split_name} edge drug indices exceed graph range.")
    if edge_df["disease_idx"].max() >= graph.num_nodes("disease"):
        raise RuntimeError(f"{split_name} edge disease indices exceed graph range.")


def load_split_bundle(graph_root, train_edge_set, valid_edge_set, test_edge_set):
    graph_root = Path(graph_root)
    train_graph, _ = load_graph(graph_root / "train_graph.bin")
    valid_graph, _ = load_graph(graph_root / "valid_graph.bin")
    test_graph, _ = load_graph(graph_root / "test_graph.bin")

    train_df = load_edge_df(graph_root / f"train_edges_{train_edge_set}.csv", "train")
    valid_df = load_edge_df(graph_root / f"valid_edges_{valid_edge_set}.csv", "valid")
    test_df = load_edge_df(graph_root / f"test_edges_{test_edge_set}.csv", "test")

    validate_eval_feasibility(valid_graph, valid_df, "valid")
    validate_eval_feasibility(test_graph, test_df, "test")

    return {
        "train_graph": train_graph,
        "valid_graph": valid_graph,
        "test_graph": test_graph,
        "train_df": train_df,
        "valid_df": valid_df,
        "test_df": test_df,
    }


def get_feature_and_metapath(graph):
    feature = {
        "drug": graph.nodes["drug"].data["h"],
        "disease": graph.nodes["disease"].data["h"],
        "CPM": graph.nodes["CPM"].data["h"],
        "CHP": graph.nodes["CHP"].data["h"],
        "gene": graph.nodes["gene"].data["h"],
    }
    metapath = ["disease_drug", "drug_disease"]
    return feature, metapath


def build_pair_tensors(edge_df, device):
    drug_idx = torch.tensor(edge_df["drug_idx"].values, dtype=torch.long, device=device)
    disease_idx = torch.tensor(edge_df["disease_idx"].values, dtype=torch.long, device=device)
    label = torch.tensor(edge_df["label"].values, dtype=torch.float32, device=device)
    return drug_idx, disease_idx, label
