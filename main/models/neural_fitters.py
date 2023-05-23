import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class TabularDataSet(Dataset):
    def __init__(self, X, y_1, y_2):
        self.X = X.copy()
        self.y1 = y_1
        self.y2 = y_2
        
    def __len__(self):
        return len(self.y1)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y1[idx], self.y2[idx]
    
class BaseMLP(nn.Module):
    def __init__(self,
                 arch,
                 input_size,
                 output_size=2,
                 dropout=0.1):
        super().__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self._set_arch(arch, input_size)
        
    def _set_arch(self, arch, input_size):
        current_size = input_size
        for lay_size in arch:
            self.layers.append(nn.Linear(current_size, lay_size))
            current_size = lay_size
            
        self.final_layer = nn.Linear(current_size, self.output_size)

    def forward(self, x):
        for lay_ in self.layers:
            x = F.relu(lay_(x))
            x = self.dropout(x)
            
        x = self.final_layer(x)
        
        return x