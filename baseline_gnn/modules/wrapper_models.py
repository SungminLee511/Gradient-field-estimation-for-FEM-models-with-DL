from torch_geometric.data import Data

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.layer import GATv2, GCN, MLP

class baseline(nn.Module):
    def __init__(
        self,
        input_dim : int,
        bc_dim : int,
        output_dim : int,
        hidden_dim : int,
        n_layers_enc : int,
        n_layers_pro : int,
        n_layers_dec : int,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.bc_dim = bc_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers_enc = n_layers_enc
        self.n_layers_pro = n_layers_pro
        self.n_layers_dec = n_layers_dec

        self.mid_dim = self.hidden_dim // 2
        self.quad_dim = self.hidden_dim // 4
        self.oct_dim = self.hidden_dim // 8

        '''
        (1) Encoder
          - default ver: diff-GNN
            a. input feature에 대해 MLP_A 통과
            b. bc feature에 대해 각각 MLP_B 통과
            c. 두 latent vector를 동일한 GNN(GCN)에 각각 통과
            d. latent vector의 차이를 계산하여 이후 processor에 사용
        '''

        self.encoder_A = MLP(
            in_channels = self.input_dim,
            hidden_channels = self.mid_dim,
            out_channels = self.hidden_dim,
            num_layers = self.n_layers_enc,
            dropout = 0.,
        )

        self.encoder_B = MLP(
            in_channels = self.input_dim + self.bc_dim,
            hidden_channels = self.mid_dim,
            out_channels = self.hidden_dim,
            num_layers = self.n_layers_enc,
            dropout = 0.,
        )

        self.encoder_GNN = GCN(
            input_dim = self.hidden_dim,
            hidden_dim = self.hidden_dim,
            output_dim = self.hidden_dim,
            num_gnn_layers = self.n_layers_enc,
            dropout = 0.,
        )

        self.encoder_ln = nn.LayerNorm(self.hidden_dim)

        '''
        (2) Processor
          - default ver: GCN
        '''

        self.processor = GCN(
                input_dim = self.hidden_dim,
                hidden_dim = self.hidden_dim,
                output_dim = self.hidden_dim,
                num_gnn_layers = self.n_layers_pro,
                residual = True,
                dropout = 0.,
            )

        # self.processor = GATv2(
        #         input_dim = self.hidden_dim,
        #         hidden_dim = self.hidden_dim,
        #         output_dim = self.hidden_dim,
        #         num_gnn_layers = self.n_layers_pro,
        #         residual = True,
        #         dropout = 0.,
        #         edge_dim = 4,
        #     )

        '''
        (3) Processor
          - default ver: MLP
        '''

        self.decoder = MLP(
            in_channels = self.hidden_dim,
            hidden_channels = self.hidden_dim,
            out_channels = self.output_dim,
            num_layers = self.n_layers_dec,
            dropout = 0.
        )

    def forward(self, graph: Data):

        # (0) data extract
        x = graph.x
        pos = graph.x[:, :3] # 좌표값
        bc = graph.bc

        edge_index = graph.edge_index
        edge_weight = graph.edge_weight

        # (1) Encoder
        x_bc = torch.cat([x, bc], dim=-1)

        x = self.encoder_A(x)
        x_bc = self.encoder_B(x_bc)

        x = self.encoder_GNN(x, edge_index, edge_weight[:, -1])
        x_bc = self.encoder_GNN(x_bc, edge_index, edge_weight[:, -1])

        x = self.encoder_ln(x)
        x_bc = self.encoder_ln(x_bc)

        x = x - x_bc

        # (2) Processor
        x = self.processor(x, edge_index, edge_weight[:, -1])

        # (3) Decoder
        y = self.decoder(x)

        return y