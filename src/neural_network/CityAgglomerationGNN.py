import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class CityAgglomerationGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(CityAgglomerationGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim // 2)
        self.classifier = nn.Linear(hidden_dim // 2, output_dim)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x, edge_index, batch=None):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        x = self.classifier(x)
        
        return x
 