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
        
    def forward(self, x, edge_index, batch=None):

        x = self.conv1(x, edge_index)
        # x = F.leaky_relu(x, negative_slope=0.01)
        x = F.silu(x)

        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = F.silu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = F.silu(x)
        
        x = self.classifier(x)
        
        x = torch.sigmoid(x) * 0.9 + 0.1
        
        
        return x
 