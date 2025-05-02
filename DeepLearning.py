import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, global_mean_pool, GCNConv, GATConv
from torch_geometric.utils import add_self_loops
import optuna

def make_graph(corr_vec, age, n_nodes, edge_thresh):
    # 1) Fisher‐transform & standardize the correlation vector
    corr_vec = np.arctanh(np.clip(corr_vec, -0.999, 0.999))
    corr_vec = (corr_vec - corr_vec.mean()) / corr_vec.std()

    # 2) Reconstruct symmetric adjacency matrix
    mat = torch.zeros((n_nodes, n_nodes), dtype=torch.float)
    iu = torch.triu_indices(n_nodes, n_nodes, offset=1)
    mat[iu[0], iu[1]] = torch.from_numpy(corr_vec).float()
    mat = mat + mat.t()

    # 3) Threshold & build edge list (including self‐loops)
    mask = mat.abs() > edge_thresh
    edge_index = mask.nonzero(as_tuple=False).t()       # [2, E]
    edge_weight = mat[mask]                            # [E]
    edge_index, edge_weight = add_self_loops(
        edge_index,
        edge_attr=edge_weight,
        fill_value=0.0,
        num_nodes=n_nodes
    )

    # 4) Compute node‐level features
    strength     = mat.abs().sum(dim=1)                # total absolute weight
    degree       = mask.sum(dim=1).float()             # count of edges
    pos_strength = mat.clamp(min=0.0).sum(dim=1)       # sum of positive weights
    neg_strength = mat.clamp(max=0.0).abs().sum(dim=1) # sum of abs(negative weights)

    # 5) Stack into feature matrix (n_nodes × 5)
    x = torch.stack([
        torch.ones(n_nodes),    # bias feature
        strength,
        degree,
        pos_strength,
        neg_strength
    ], dim=1)

    # 6) Return PyG Data object
    return Data(
        x          = x,
        edge_index = edge_index,
        edge_attr  = edge_weight,
        y          = torch.tensor([age], dtype=torch.float)
    )


def vec_to_corrmat(corr_vec, n_nodes=200):
    """
    Turn the flattened upper-triangle (length 19 900) back into a full
    n_nodes × n_nodes symmetric matrix.

    Parameters
    ----------
    corr_vec : 1-D array-like, shape (n_edges,)
        Upper-triangle values, **excluding** the diagonal, ordered row-wise
        (i<j).
    n_nodes : int, default 200
        How many ROIs (so n_edges = n_nodes*(n_nodes-1)//2 must match).

    Returns
    -------
    mat : ndarray, shape (n_nodes, n_nodes)
        Symmetric correlation matrix with zeros on the diagonal.
    """
    mat = np.zeros((n_nodes, n_nodes), dtype=float)
    iu  = np.triu_indices(n_nodes, k=1)      # i<j indices
    mat[iu] = corr_vec                       # fill upper triangle
    mat += mat.T                             # make symmetric
    return mat

class AgeGCN(torch.nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(5, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin   = torch.nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x).squeeze()


class AgeGAT(torch.nn.Module):
    def __init__(self, in_dim=5, hidden_dim=32, heads1=4, heads2=1, attn_dropout=0.6):
        super().__init__()
        # First GAT layer
        self.conv1 = GATConv(
            in_channels   = in_dim,
            out_channels  = hidden_dim,
            heads         = heads1,
            concat        = True,      # outputs hidden_dim * heads1
            dropout       = attn_dropout
        )
        # Second GAT layer
        self.conv2 = GATConv(
            in_channels   = hidden_dim * heads1,
            out_channels  = hidden_dim,
            heads         = heads2,
            concat        = False,     # outputs hidden_dim
            dropout       = attn_dropout
        )
        # Final MLP to scalar age
        self.lin = torch.nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, edge_weight, batch = (
        data.x, data.edge_index, data.edge_attr, data.batch
    )

        # 1) Attention convolution 1
        x = F.elu(self.conv1(x, edge_index, edge_weight))
        # 2) Attention convolution 2
        x = F.elu(self.conv2(x, edge_index, edge_weight))
        # 3) Global pooling to get graph embedding
        x = global_mean_pool(x, batch)
        # 4) Final regression head
        return self.lin(x).squeeze()

class AgeSAGE(torch.nn.Module):
    def __init__(self, in_dim=3, hid=32):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hid)
        self.bn1   = torch.nn.BatchNorm1d(hid)
        self.conv2 = SAGEConv(hid, hid)
        self.bn2   = torch.nn.BatchNorm1d(hid)
        self.lin   = torch.nn.Linear(hid, 1)

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        x = F.dropout(F.relu(self.bn1(self.conv1(x, ei))), p=0.4, training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x, ei))), p=0.4, training=self.training)
        x = global_mean_pool(x, batch)
        return self.lin(x).squeeze()

class CorrCAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        # ---------- encoder ----------
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),   # → 32×100×100
            nn.ReLU(), 
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # → 64×50×50
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # → 128×25×25
            nn.ReLU(),
            # optional extra:
            nn.Conv2d(128, 256, 3, stride=2, padding=1),# → 256×13×13
            nn.ReLU(),
        )
        self.flatten   = nn.Flatten()
        self.fc_mu     = nn.Linear(256 * 13 * 13, latent_dim)

        # ---------- decoder ----------
        self.fc_up     = nn.Linear(latent_dim, 256 * 13 * 13)
        self.dec   = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64,  3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,  32,  3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32,   1,  3, stride=2, padding=1, output_padding=1),
        )

    def encode(self, x):
        x = self.enc(x)
        x = self.flatten(x)
        return self.fc_mu(x)

    def decode(self, z):
        x = self.fc_up(z).view(-1, 256, 13, 13)
        return self.dec(x)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


class ConditionCAE(nn.Module):
    def __init__(self, latent_dim=64, extra_dim=11):
        super().__init__()
        # ---------- encoder ----------
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),    # → 32×100×100
            nn.ReLU(), 
            nn.Conv2d(32, 64, 3, stride=2, padding=1),   # → 64×50×50
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # → 128×25×25
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), # → 256×13×13
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        # now the FC sees both the flattened maps *and* the extra features
        self.fc_mu = nn.Linear(256 * 13 * 13 + extra_dim, latent_dim)

        # ---------- decoder ----------
        self.fc_up = nn.Linear(latent_dim, 256 * 13 * 13)
        self.dec   = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64,  3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,  32,  3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32,   1,  3, stride=2, padding=1, output_padding=1),
        )

    def encode(self, x, extra):
        # x: (batch,1,200,200), extra: (batch, extra_dim)
        x_enc   = self.enc(x)                    # → (batch,256,13,13)
        x_flat  = self.flatten(x_enc)            # → (batch, 256*13*13)
        x_cat   = torch.cat([x_flat, extra], dim=1)
        return self.fc_mu(x_cat)                 # → (batch, latent_dim)

    def decode(self, z):
        x_up = self.fc_up(z).view(-1, 256, 13, 13)
        return self.dec(x_up)                    # → (batch,1,200,200)

    def forward(self, x, extra):
        z     = self.encode(x, extra)
        recon = self.decode(z)
        return recon, z

class AgeMLP(torch.nn.Module):
    def __init__(self, in_dim=256, hidden=64):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_dim, hidden)
        self.fc2 = torch.nn.Linear(hidden, hidden)
        self.fc3 = torch.nn.Linear(hidden, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x).squeeze()


class AgeMLP2(torch.nn.Module):
    def __init__(self, in_dim, hidden1=256, drop=0.2):
        super().__init__()
        self.fc1  = torch.nn.Linear(in_dim, hidden1)
        self.bn1  = torch.nn.BatchNorm1d(hidden1)
        self.drop1 = torch.nn.Dropout(drop)
        self.fc2  = torch.nn.Linear(hidden1, hidden1)
        self.bn2  = torch.nn.BatchNorm1d(hidden1)
        self.drop2 = torch.nn.Dropout(drop)
        self.fc3  = torch.nn.Linear(hidden1, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        return self.fc3(x).squeeze()

def objective(trial, train_loader, val_loader, model_cls, num_epochs=50):
    # 1) Hyperparameter suggestions
    hidden_dim   = trial.suggest_int("hidden_dim", 16, 128, log=True)
    lr           = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    wd           = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)

    # 2) Model & optimizer
    model     = model_cls(in_dim=256, hidden1 = hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # 3) Early stopping setup
    best_val      = float("inf")
    patience      = 0
    max_patience  = 5  # stop if no improvement for 5 epochs

    # 4) Training loop with validation & pruning
    for epoch in range(1, num_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = F.mse_loss(pred, yb.view(-1))
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                val_losses.append(F.mse_loss(pred, yb.view(-1)).item())
        val_mse = sum(val_losses) / len(val_losses)

        # Report to Optuna for pruning
        trial.report(val_mse, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Early stopping logic
        if val_mse < best_val:
            best_val = val_mse
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                break  # stop training early

    return best_val