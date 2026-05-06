import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):

    def __init__(
        self,
        in_channels:     int   = 164,
        hidden_channels: int   = 128,
        out_channels:    int   = 1,
        dropout:         float = 0.5,
    ):
        super().__init__()
        self.dropout = dropout

        # Three conv layers
        self.conv0 = GCNConv(in_channels,     hidden_channels)
        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        self.bn0 = nn.BatchNorm1d(hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        # Project raw input to hidden_dim once — used for the layer-0 residual.
        # No bias: this is a pure linear re-scaling of the feature space.
        self.input_proj = nn.Linear(in_channels, hidden_channels, bias=False)

        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # ── Layer 0 ───────────────────────────────────────────────────────────
        h = F.relu(self.bn0(self.conv0(x, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + self.input_proj(x)          # residual: project raw features

        # ── Layer 1 ───────────────────────────────────────────────────────────
        h1 = F.relu(self.bn1(self.conv1(h, edge_index)))
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        h1 = h1 + h                          # residual: add previous embedding

        # ── Layer 2 ───────────────────────────────────────────────────────────
        h2 = F.relu(self.bn2(self.conv2(h1, edge_index)))
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        h2 = h2 + h1                         # residual: add previous embedding

        return self.classifier(h2)

    def embed(self, x, edge_index):
        """Return pre-classifier node embeddings (for explainability)."""
        with torch.no_grad():
            h  = F.relu(self.bn0(self.conv0(x, edge_index))) + self.input_proj(x)
            h1 = F.relu(self.bn1(self.conv1(h,  edge_index))) + h
            h2 = F.relu(self.bn2(self.conv2(h1, edge_index))) + h1
        return h2
