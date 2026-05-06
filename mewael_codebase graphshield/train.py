import torch
from src.data.preprocess import load_snapshots
from src.models.gcn import GCN
from src.models.graphsage import GraphSAGE
from src.training.trainer import train_epoch, evaluate

MODEL      = "gcn"
EPOCHS     = 50
LR         = 1e-3
HIDDEN     = 128
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading snapshots...")
snapshots = load_snapshots()
train_snaps = [s for s in snapshots if s.timestep <= 34]
val_snaps   = [s for s in snapshots if 35 <= s.timestep <= 39]
test_snaps  = [s for s in snapshots if s.timestep >= 40]

n_licit   = sum((s.y[s.train_mask] == 0).sum().item() for s in train_snaps)
n_illicit = sum((s.y[s.train_mask] == 1).sum().item() for s in train_snaps)
pos_weight = torch.tensor([n_licit / n_illicit]).to(DEVICE)
print(f"pos_weight: {pos_weight.item():.2f}  (licit={n_licit}, illicit={n_illicit})")

if MODEL == "gcn":
    model = GCN(in_channels=165, hidden_channels=HIDDEN, out_channels=1)
else:
    model = GraphSAGE(in_channels=165, hidden_channels=HIDDEN, out_channels=1)

model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# --- Training loop ---
print(f"\nTraining {MODEL.upper()} for {EPOCHS} epochs on {DEVICE}\n")
for epoch in range(1, EPOCHS + 1):
    loss = train_epoch(model, train_snaps, optimizer, pos_weight, DEVICE)
    if epoch % 5 == 0:
        val_metrics = evaluate(model, val_snaps, DEVICE)
        print(f"Epoch {epoch:3d} | loss {loss:.4f} | "
              f"val F1 {val_metrics['f1']:.4f} | "
              f"val PR-AUC {val_metrics['pr_auc']:.4f}")

test_metrics = evaluate(model, test_snaps, DEVICE)
print(f"\nTest results:")
print(f"  F1(illicit): {test_metrics['f1']:.4f}")
print(f"  ROC-AUC:     {test_metrics['roc_auc']:.4f}")
print(f"  PR-AUC:      {test_metrics['pr_auc']:.4f}")
