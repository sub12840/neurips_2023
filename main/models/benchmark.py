import torch
import torch.nn as nn
import torch.nn.functional as F

class OnetoNet(nn.Module):
    def __init__(self,
                 architecture: int | list, 
                 input_dim: int,
                 n_tasks: int,
                 activation='relu', 
                 dropout=0.1,
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.n_tasks = n_tasks
        self.layers = nn.ModuleDict()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        self._set_architecture(architecture)

    def _set_architecture(self, architecture):
        if isinstance(architecture, int):
            self.layers['embedding'] = nn.Linear(self.input_dim, architecture)
            self.embedding_dim = architecture

        if isinstance(architecture, list):
            raise NotImplementedError
        
        # Now create V matrix
        self.layers['output'] = nn.Linear(self.embedding_dim, self.n_tasks)

    def forward(self, x):
        W = self.layers['embedding'](x)
        if self.activation is not None:
            W = F.relu(W)
            W = self.dropout(W)

        V = self.layers['output'](W)

        return W, V
    
class MMDLoss(nn.Module):
    def __init__(self, 
                 kernel='gaussian', 
                 mu=1):
        super().__init__()
        self._set_kernel(kernel, mu)

    def _set_kernel(self, kernel, mu):
        if kernel == 'quadratic':
            raise NotImplementedError
        elif kernel == 'gaussian':
            assert isinstance(mu, float), f"you need to provide a valid mu parameter \
                                               you provided {type(mu)}"
            self.kernel_ = self._gaussian_kernel
            self.mu = mu

    def forward(self, x, y):
        x_dist = torch.cdist(x,x).reshape(-1,)
        y_dist = torch.cdist(y,y).reshape(-1,)
        x_y_dist = torch.cdist(x,y).reshape(-1,)

        x_ker = torch.exp(-0.5*x_dist)
        y_ker = torch.exp(-0.5*y_dist)
        x_y_ker = torch.exp(-0.5*x_y_dist)

        total_dists = x_ker.mean() + y_ker.mean() - 2*x_y_ker.mean()

        return total_dists


    def _gaussian_kernel(self, dist_vector):
        return dist_vector.mul(-0.5 * (1/self.mu)).exp()
