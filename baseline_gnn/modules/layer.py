import torch
import torch.nn as nn
from typing import Optional
from torch.nn import Identity
from torch_geometric.nn import GCNConv, GATv2Conv, LayerNorm, InstanceNorm, GraphNorm, BatchNorm
import torch.nn.functional as F

def get_gnn_norm(norm_type):
    if norm_type in [ 'layer', 'layernorm' ,  'LayerNorm']:
        return LayerNorm
    elif norm_type in [ 'instance', 'instancenorm' ,  'InstanceNorm']:
        return InstanceNorm
    
    elif norm_type in [ 'batch', 'batchnorm' ,  'BatchNorm']:
        return BatchNorm
    
    elif norm_type in ['graph', 'graphnorm', 'GraphNorm']:
        return GraphNorm
    else :
        return Identity


class GATv2(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 num_gnn_layers: int,
                 residual: bool = False,
                 norm_type: str = 'layer',
                 activation_fn: nn.Module = nn.GELU(),
                 dropout: float = 0.0,
                 edge_dim: int = None,
                 n_heads: int = 4):
        super(GATv2, self).__init__()
        self.num_gnn_layers = num_gnn_layers
        self.hidden_dim = hidden_dim

        self.min_dim = int(hidden_dim / n_heads)
        norm_layer = get_gnn_norm(norm_type)
        
        self.conv_layers = nn.ModuleList()
        self.gnn_norm_layers = nn.ModuleList()
        
        if num_gnn_layers == 1:
            self.conv_layers.append(
                GATv2Conv(input_dim, int(output_dim / n_heads), heads=n_heads, edge_dim=edge_dim, concat=True, dropout=dropout)
                )
            self.gnn_norm_layers.append(norm_layer(output_dim))
        else:
            self.conv_layers.append(
                GATv2Conv(input_dim, self.min_dim, heads=n_heads, edge_dim=edge_dim, concat=True, dropout=dropout)
                )
            self.gnn_norm_layers.append(norm_layer(hidden_dim))
            for _ in range(num_gnn_layers - 2):
                self.conv_layers.append(
                    GATv2Conv(hidden_dim, self.min_dim, heads=n_heads, edge_dim=edge_dim, concat=True, dropout=dropout)
                    )
                self.gnn_norm_layers.append(norm_layer(hidden_dim))
            self.conv_layers.append(
                GATv2Conv(hidden_dim, int(output_dim / n_heads), heads=n_heads, edge_dim=edge_dim, concat=True, dropout=dropout)
                )
            self.gnn_norm_layers.append(norm_layer(output_dim))
            
        self.activation_fn = activation_fn
        self.residual = residual

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None) -> torch.Tensor:
        for i in range(self.num_gnn_layers):
            x_res = x
            x = self.conv_layers[i](x, edge_index, edge_attr)
            if self.residual:
                x = x + x_res
            x = self.gnn_norm_layers[i](x)
            x = self.activation_fn(x)
        return x


class GCN(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 num_gnn_layers: int,
                 residual: bool = False,
                 norm_type: str = 'layer',
                 activation_fn: nn.Module = nn.GELU(),
                 dropout: float = 0.0):
        super(GCN, self).__init__()
        self.num_gnn_layers = num_gnn_layers
        norm_layer = get_gnn_norm(norm_type)
        
        self.conv_layers = nn.ModuleList()
        self.gnn_norm_layers = nn.ModuleList()
        
        if num_gnn_layers == 1:
            self.conv_layers.append(
                GCNConv(in_channels=input_dim, out_channels=output_dim, add_self_loops=True)
            )
            self.gnn_norm_layers.append(norm_layer(output_dim))
        else:
            self.conv_layers.append(
                GCNConv(in_channels=input_dim, out_channels=hidden_dim, add_self_loops=True)
            )
            self.gnn_norm_layers.append(norm_layer(hidden_dim))
            for _ in range(num_gnn_layers - 2):
                self.conv_layers.append(
                    GCNConv(in_channels=hidden_dim, out_channels=hidden_dim, add_self_loops=True)
                )
                self.gnn_norm_layers.append(norm_layer(hidden_dim))
            self.conv_layers.append(
                GCNConv(in_channels=hidden_dim, out_channels=output_dim, add_self_loops=True)
            )
            self.gnn_norm_layers.append(norm_layer(output_dim))
            
        self.activation_fn = activation_fn
        self.residual = residual

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None) -> torch.Tensor:
        for i in range(self.num_gnn_layers):
            x_res = x
            x = self.conv_layers[i](x, edge_index, edge_weight)
            if self.residual:
                x = x + x_res
            x = self.gnn_norm_layers[i](x)
            x = self.activation_fn(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int, dropout: float = 0.0):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.lins = nn.ModuleList()
        
        if num_layers == 1:
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x