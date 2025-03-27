import torch
import torch.nn as nn
from torch.nn import functional as F


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        sigmoid_output: bool = False,
        affine_func=nn.Linear,
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            affine_func(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x: torch.Tensor):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

class ResidualMLP(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_mlp,
        num_layer_per_mlp,
        sigmoid_output: bool = False,
        affine_func=nn.Linear,
    ):
        super().__init__()
        self.num_mlp = num_mlp
        self.in2hidden_dim = affine_func(input_dim, hidden_dim)
        self.hidden2out_dim = affine_func(hidden_dim, output_dim)
        self.mlp_list = nn.ModuleList(
            MLP(
                hidden_dim,
                hidden_dim,
                hidden_dim,
                num_layer_per_mlp,
                affine_func=affine_func,
            ) for _ in range(num_mlp)
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x: torch.Tensor):
        x = self.in2hidden_dim(x)
        for mlp in self.mlp_list:
            out = mlp(x)
            x = x + out
        out = self.hidden2out_dim(x)
        return out