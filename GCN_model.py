import torch
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from torch.nn import L1Loss
import numpy as np


class GCN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 64)
        self.lin = torch.nn.Linear(64, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)

        x = self.lin(x)
        return x

    def fit(self, train_loader, val_loader, optimizer, criterion, epochs):
        mae_criterion = L1Loss()
        for epoch in range(epochs):
            self.train()
            for data in train_loader:
                optimizer.zero_grad()
                output = self(data)
                loss = criterion(output, data.y.view_as(output))
                mae_loss = mae_criterion(output, data.y.view_as(output))
                loss.backward()
                optimizer.step()

            val_loss, val_mae_loss = self.validate(val_loader, criterion)  # Предполагая, что val_loader определён
            print(
                f'Epoch {epoch + 1}, MSE Loss: {loss.item()}, Val MAE Loss: {val_mae_loss}, MAE Loss: {mae_loss.item()}, Val MSE Loss: {val_loss}')

    def validate(self, val_loader, criterion):
        self.eval()
        total_loss = 0
        total_mae_loss = 0
        mae_criterion = L1Loss()
        with torch.no_grad():
            for data in val_loader:
                output = self(data)
                loss = criterion(output, data.y.view_as(output))
                mae_loss = mae_criterion(output, data.y.view_as(output))
                total_loss += loss.item()
                total_mae_loss += mae_loss.item()
        return total_loss / len(val_loader), total_mae_loss / len(val_loader)

    def predict(self, test_loader):
        self.eval()
        predictions = []
        with torch.no_grad():
            for data in test_loader:
                output = self(data)
                predictions.append(output.squeeze().cpu().numpy())
        print(predictions)
        predictions = np.array(predictions).reshape(-1, 1)
        return predictions
