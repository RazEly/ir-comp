"""
Student implementation file — implement all TODO sections below.

This is the only Section B file you should submit.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.nn import SAGEConv


def get_feature_vectors(nodes_df):
    """Return node feature matrix x as a float torch.Tensor."""
    features = [
        np.array(feature_str.strip("[]").split(","), dtype=np.float32)
        for feature_str in nodes_df["features"].to_numpy()
    ]

    x = np.vstack(features)
    return torch.tensor(x, dtype=torch.float32)


def get_edges(edges_df, inverse_node_id_mapping):
    """Return edge_index as a long torch.Tensor of shape [2, num_edges]."""
    source = [
        inverse_node_id_mapping[int(node_id)]
        for node_id in edges_df["sourceNodeId"].to_numpy()
    ]

    target = [
        inverse_node_id_mapping[int(node_id)]
        for node_id in edges_df["targetNodeId"].to_numpy()
    ]

    edge_index = np.vstack([source, target]).astype(np.int64)
    return torch.tensor(edge_index, dtype=torch.long)


def get_labels(nodes_df, subject_mapping):
    """Return node labels y as a long torch.Tensor."""
    labels = nodes_df["subject"].map(subject_mapping).to_numpy(dtype=np.int64)
    return torch.tensor(labels, dtype=torch.long)


class GraphSAGE(torch.nn.Module):
    def __init__(self, hidden_channels, output_dim, seed):
        super().__init__()

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # In run.py, the first argument passed here is data.x.shape[1].
        # So despite the template name, hidden_channels is actually input_dim.
        input_dim = hidden_channels
        hidden_dim = 64

        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)

        self.dropout = 0.5

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = torch.nn.functional.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )
        x = self.conv2(x, edge_index)
        return x


def train(data, model, optimizer, epochs, evaluate_fn):
    """
    Train the model for the given number of epochs.

    Use validation accuracy to track the best checkpoint.
    Save the best full model object to 'best_model.pt'.
    """
    best_valid_acc = -1.0
    best_valid_loss = float("inf")

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()

        logits = model(data.x, data.edge_index)

        loss = torch.nn.functional.cross_entropy(
            logits[data.train_mask],
            data.y[data.train_mask],
        )

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_logits = model(data.x, data.edge_index)

            valid_loss = torch.nn.functional.cross_entropy(
                valid_logits[data.valid_mask],
                data.y[data.valid_mask],
            ).item()

        valid_acc = evaluate_fn(model, data, data.valid_mask)

        if valid_acc > best_valid_acc or (
            valid_acc == best_valid_acc and valid_loss < best_valid_loss
        ):
            best_valid_acc = valid_acc
            best_valid_loss = valid_loss

            checkpoint_path = __file__[: -len("gnn.py")] + "best_model.pt"
            torch.save(model, checkpoint_path)

