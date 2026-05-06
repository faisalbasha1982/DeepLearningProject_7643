import torch
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

def train_epoch(model, snapshots, optimizer, pos_weight, device):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    total_loss = 0

    for snap in snapshots:
        x          = snap.x.to(device)
        edge_index = snap.edge_index.to(device)
        y          = snap.y[snap.train_mask].float().to(device)

        logits = model(x, edge_index)[snap.train_mask.to(device)]
        loss   = criterion(logits.squeeze(), y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(snapshots)


@torch.no_grad()
def evaluate(model, snapshots, device):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    for snap in snapshots:
        x          = snap.x.to(device)
        edge_index = snap.edge_index.to(device)
        mask       = snap.train_mask

        logits = model(x, edge_index)[mask.to(device)].squeeze().cpu()
        probs  = torch.sigmoid(logits)
        preds  = (probs >= 0.5).long()
        labels = snap.y[mask]

        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    f1     = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
    roc    = roc_auc_score(all_labels, all_probs)
    pr_auc = average_precision_score(all_labels, all_probs)

    return {"f1": f1, "roc_auc": roc, "pr_auc": pr_auc}



def train_epoch_temporal(model, snapshots, optimizer, pos_weight, device, chunk_size=8):
  
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    total_loss = 0
    n_chunks   = 0
    state      = model.initial_state()

    for start in range(0, len(snapshots), chunk_size):
        chunk = snapshots[start : start + chunk_size]

        state = state.detach()

        all_logits, all_masks, state = model(chunk, device, state=state)

        chunk_loss = 0
        for logits, mask, snap in zip(all_logits, all_masks, chunk):
            y = snap.y[mask].float().to(device)
            chunk_loss += criterion(logits[mask.to(device)], y)

        avg = chunk_loss / len(chunk)
        optimizer.zero_grad()
        avg.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += avg.item()
        n_chunks   += 1

    return total_loss / n_chunks


@torch.no_grad()
def evaluate_temporal(model, context_snaps, eval_snaps, device):
    
    model.eval()

    _, _, state = model(context_snaps, device)

    all_logits, all_masks, _ = model(eval_snaps, device, state=state)

    all_preds, all_probs, all_labels = [], [], []
    for logits, mask, snap in zip(all_logits, all_masks, eval_snaps):
        probs  = torch.sigmoid(logits[mask.to(device)]).cpu()
        preds  = (probs >= 0.5).long()
        labels = snap.y[mask]

        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    f1     = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
    roc    = roc_auc_score(all_labels, all_probs)
    pr_auc = average_precision_score(all_labels, all_probs)

    return {"f1": f1, "roc_auc": roc, "pr_auc": pr_auc}

train_epoch_evolve = train_epoch_temporal
evaluate_evolve    = evaluate_temporal
