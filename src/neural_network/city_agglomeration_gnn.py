import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch
class CityAgglomerationGNN(nn.Module):
    """
    O retea neuronala de grafuri (GNN) pentru analiza aglomerarilor urbane.
    
    Acest model foloseste Retele Convolutionale de Grafuri (GCN) pentru a procesa
    date structurate sub forma de graf reprezentand orase si conexiunile acestora.
    Utilizeaza o arhitectura GCN cu trei straturi cu activari ReLU si dropout pentru regularizare.
    
    Modelul este conceput pentru sarcini legate de predictia aglomerarilor urbane,
    cum ar fi identificarea oraselor care apartin aceleiasi zone metropolitane
    sau predictia modelelor de conectivitate urbana.
    
    Arhitectura:
        - Stratul de intrare: GCNConv (input_dim -> hidden_dim)
        - Stratul ascuns 1: GCNConv (hidden_dim -> hidden_dim)
        - Stratul ascuns 2: GCNConv (hidden_dim -> hidden_dim // 2)
        - Stratul de iesire: Linear (hidden_dim // 2 -> output_dim)
        - Dropout (p=0.3) aplicat dupa primele doua straturi GCN
        - Activare ReLU dupa fiecare strat GCN, exceptand ultimul
    
    Args:
        input_dim (int): Numarul caracteristicilor de intrare per nod
        hidden_dim (int, optional): Numarul unitatilor ascunse in straturile GCN. Implicit 64.
        output_dim (int, optional): Numarul caracteristicilor de iesire. Implicit 1.
    """
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(CityAgglomerationGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim // 2)
        self.classifier = nn.Linear(hidden_dim // 2, output_dim)
        self.dropout = nn.Dropout(0.3)


    # def forward(self, x, edge_index, batch=None):

    #     # In your forward pass, add skip connections:
    #     x = self.conv1(x, edge_index)
    #     x = F.leaky_relu(x)
    #     x = self.dropout(x)

    #     x = self.conv2(x, edge_index)
    #     x = F.leaky_relu(x)
    #     x = self.dropout(x)

    #     x = self.conv3(x, edge_index)
    #     x = F.leaky_relu(x)
        
    #     output = self.classifier(x)
        
    #     return output
        
    def forward(self, x, edge_index, batch=None):

        # In your forward pass, add skip connections:
        x1 = self.conv1(x, edge_index)
        x1 = F.leaky_relu(x1)
        x1 = self.dropout(x1)

        x2 = self.conv2(x1, edge_index)
        x2 = F.leaky_relu(x2)
        x2 = self.dropout(x2)
        x2 = x2 + x1  # Residual connection

        x3 = self.conv3(x2, edge_index)
        x3 = F.leaky_relu(x3)
        
        # Add a simple skip connection from input
        skip_pred = torch.mean(x, dim=1, keepdim=True) * 0.5
        main_pred = self.classifier(x3)
        output = 0.9 * main_pred + 0.1 * skip_pred
        
        
        return output
 