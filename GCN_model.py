import torch
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)
        self.lin = torch.nn.Linear(num_classes, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)

        x = self.lin(x)
        return torch.sigmoid(x)

    def fit(self, train_loader, val_loader, optimizer, criterion, epochs):
        for epoch in range(epochs):
            self.train()
            for data in train_loader:
                optimizer.zero_grad()
                output = self(data)
                loss = criterion(output.squeeze(), data.y.type_as(output))
                loss.backward()
                optimizer.step()

            val_loss = self.validate(val_loader, criterion)  # Предполагая, что val_loader определён
            print(f'Epoch {epoch + 1}, Loss: {loss.item()}, Val Loss: {val_loss}')

    def validate(self, val_loader, criterion):
        self.eval()
        total_loss = 0
        with torch.no_grad():
            for data in val_loader:
                output = self(data)
                loss = criterion(output.squeeze(), data.y.type_as(output))
                total_loss += loss.item()
        return total_loss / len(val_loader)

    def predict(self, test_loader):
        self.eval()
        predictions = []
        with torch.no_grad():
            for data in test_loader:
                output = self(data)
                predictions.append(output.squeeze().cpu().numpy())
        return predictions