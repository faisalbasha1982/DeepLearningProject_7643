import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class TGN(nn.Module):

    def __init__(
        self,
        in_channels  = 164,   # 164 features (timestep col dropped)
        memory_dim   = 128,   # global memory vector size
        hidden_dim   = 128,   # hidden embedding dimension
        heads        = 4,     # number of attention heads in GAT
        dropout      = 0.3,
    ):
        super().__init__()

        self.memory_dim = memory_dim
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(in_channels + memory_dim, hidden_dim)

        self.input_ln = nn.LayerNorm(hidden_dim)

        # ── Graph Attention layer ─────────────────────────────────────────────
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"
        self.gat = GATConv(
            in_channels  = hidden_dim,
            out_channels = hidden_dim // heads,
            heads        = heads,
            dropout      = dropout,
            concat       = True,
        )
        self.gat_ln  = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.memory_updater = nn.GRUCell(
            input_size  = hidden_dim,
            hidden_size = memory_dim,
        )

        self.classifier = nn.Linear(hidden_dim, 1)

    def initial_state(self):
        """Zero memory — used at t=0 before any snapshot has been seen."""
        device = next(self.parameters()).device
        return torch.zeros(self.memory_dim, device=device)


    def forward(self, snapshots, device, state=None):
        """
        snapshots : time-ordered list of PyG Data objects
        state     : global memory tensor from end of previous call

        Returns (logits_list, masks_list, final_memory)
        """
        memory = state if state is not None else self.initial_state()

        all_logits = []
        all_masks  = []

        for snap in snapshots:
            x          = snap.x.to(device)           # (N, 164)
            edge_index = snap.edge_index.to(device)
            N          = x.size(0)

            
            mem  = memory.unsqueeze(0).expand(N, -1)          
            x_in = torch.cat([x, mem], dim=1)                 
            h    = F.relu(self.input_ln(self.input_proj(x_in)))  

            h = F.relu(self.gat_ln(self.gat(h, edge_index)))  # (N, hidden_dim)
            h = self.dropout(h)

            logits = self.classifier(h).squeeze(-1)            # (N,)
            all_logits.append(logits)
            all_masks.append(snap.train_mask)

            graph_rep = h.mean(dim=0)                   
            memory = self.memory_updater(
                graph_rep.unsqueeze(0),     
                memory.unsqueeze(0),        
            ).squeeze(0)                                     

        return all_logits, all_masks, memory
