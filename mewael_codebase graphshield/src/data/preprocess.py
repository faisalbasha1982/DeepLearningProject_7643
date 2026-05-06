import pandas as pd
import torch
from torch_geometric.data import Data

# 1 = illicit, 0 = licit, -1 = unknown (masked out during training)
LABEL_MAP = {"1": 1, "2": 0, "unknown": -1}


def load_snapshots(raw_dir="data/raw"):
    # Load the three raw files
    features = pd.read_csv(f"{raw_dir}/elliptic_txs_features.csv", header=None)
    classes  = pd.read_csv(f"{raw_dir}/elliptic_txs_classes.csv")
    edges    = pd.read_csv(f"{raw_dir}/elliptic_txs_edgelist.csv")

    features.columns = ["txId", "timestep"] + [f"feat_{i}" for i in range(1, 166)]

    # Attach labels
    df = features.merge(classes, on="txId", how="left").copy()
    df["y"] = df["class"].map(LABEL_MAP).fillna(-1).astype(int)
    
    snapshots = []

    for t in sorted(df["timestep"].unique()):
        nodes = df[df["timestep"] == t].reset_index(drop=True)

        # Map txId → local row index for this timestep
        id_to_idx = {txid: i for i, txid in enumerate(nodes["txId"])}

        x = torch.tensor(nodes[feat_cols].values, dtype=torch.float)
        y = torch.tensor(nodes["y"].values, dtype=torch.long)

        t_edges = edges[edges["txId1"].isin(id_to_idx) & edges["txId2"].isin(id_to_idx)]
        src = [id_to_idx[n] for n in t_edges["txId1"]]
        dst = [id_to_idx[n] for n in t_edges["txId2"]]

        snapshots.append(Data(
            x=x,
            edge_index=torch.tensor([src, dst], dtype=torch.long),
            y=y,
            train_mask=(y != -1),
            timestep=int(t)
        ))

    return snapshots


if __name__ == "__main__":
    snapshots = load_snapshots()

    train = [s for s in snapshots if s.timestep <= 34]
    val   = [s for s in snapshots if 35 <= s.timestep <= 39]
    test  = [s for s in snapshots if s.timestep >= 40]

    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    s = train[0]
    print(f"t={s.timestep}: {s.num_nodes} nodes, {s.edge_index.shape[1]} edges, "
          f"{s.train_mask.sum().item()} labeled")
