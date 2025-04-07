import os 
import numpy as np
import wandb
import hydra
import h5py
import yaml

from torchinfo import summary
from omegaconf import DictConfig

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GraphLoader
from torch_geometric.utils import to_undirected, add_self_loops

from modules import *
from laplacian import *



class Trainer():
    def __init__(self, cfg: DictConfig, shuffle=True):
        self.device = cfg.experiment.device
        self.max_epochs = cfg.dataset.max_epochs

        self.E =113.8e9
        self.nu = 0.342
        self.rho = 4.47e-3

        self.ckpt_path = "/home/ubuntu/SML/SimJEB/laplacian/gatv2/ckpt"
                
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)

        train_graph_list, valid_graph_list = self.process_data(cfg)

        self.train_loader = GraphLoader(train_graph_list, batch_size=1, shuffle=shuffle) # default는 True
        self.valid_loader = GraphLoader(valid_graph_list, batch_size=1, shuffle=False)  

        self.make_model_components(cfg)
        
        self.min_train_loss = 1e+20
        self.min_val_loss = 1e+20

        self.min_train_epoch = 0
        self.min_val_epoch = 0


    def process_data(self, cfg):
        train_graph_list = []
        valid_graph_list = []

        train_sample_id = cfg.dataset.train_sample_id
        valid_sample_id = cfg.dataset.valid_sample_id

        for sample_idx in train_sample_id:

            with h5py.File(f"/data/SimJEB/laplacian/{sample_idx}.h5", 'r') as f:
                x = torch.tensor(f['x'][:])
                bc = torch.tensor(f['bc'][:])
                edge_index = torch.tensor(f['edge_index'][:])
                edge_weight = torch.tensor(f['edge_weight'][:]).unsqueeze(-1)
                y = torch.tensor(f['y'][:])

                edge_rbe2 = torch.tensor(f['edge_rbe2'][:])
                edge_rbe3 = torch.tensor(f['edge_rbe3'][:])

                stress = torch.tensor(f['stress'][:])
                tetra_edge_id = torch.tensor(f['tetra_edge_id'][:])
                tetra_edge_orientation = torch.tensor(f['tetra_edge_orientation'][:])
                coords = torch.tensor(f['coords'][:])
                elements = torch.tensor(f['elements'][:])
                edge = torch.tensor(f['edge'][:])
                displacement = torch.tensor(f['displacement'][:])
                element_stress_tensor = torch.tensor(f['element_stress_tensor'][:])
                element_vm_stress = torch.tensor(f['element_vm_stress'][:])


            train_graph_list.append(Data(x = x, 
                                            bc = bc, 
                                            edge_index = edge_index,
                                            edge_weight = edge_weight, 
                                            y = y,

                                            edge_rbe2 = edge_rbe2,
                                            edge_rbe3 = edge_rbe3,

                                            stress = stress,
                                            tetra_edge_id = tetra_edge_id,
                                            tetra_edge_orientation = tetra_edge_orientation,
                                            coords = coords,
                                            elements = elements,
                                            edge = edge,
                                            displacement = displacement,
                                            element_stress_tensor = element_stress_tensor,
                                            element_vm_stress = element_vm_stress,
                                            E = self.E,
                                            nu = self.nu,
                                            rho = self.rho))



        for sample_idx in valid_sample_id:

            with h5py.File(f"/data/SimJEB/laplacian/{sample_idx}.h5", 'r') as f:
                x = torch.tensor(f['x'][:])
                bc = torch.tensor(f['bc'][:])
                edge_index = torch.tensor(f['edge_index'][:])
                edge_weight = torch.tensor(f['edge_weight'][:]).unsqueeze(-1)
                y = torch.tensor(f['y'][:])

                edge_rbe2 = torch.tensor(f['edge_rbe2'][:])
                edge_rbe3 = torch.tensor(f['edge_rbe3'][:])

                stress = torch.tensor(f['stress'][:])
                tetra_edge_id = torch.tensor(f['tetra_edge_id'][:])
                tetra_edge_orientation = torch.tensor(f['tetra_edge_orientation'][:])
                coords = torch.tensor(f['coords'][:])
                elements = torch.tensor(f['elements'][:])
                edge = torch.tensor(f['edge'][:])
                displacement = torch.tensor(f['displacement'][:])
                element_stress_tensor = torch.tensor(f['element_stress_tensor'][:])
                element_vm_stress = torch.tensor(f['element_vm_stress'][:])


            valid_graph_list.append(Data(x = x, 
                                            bc = bc, 
                                            edge_index = edge_index,
                                            edge_weight = edge_weight, 
                                            y = y,

                                            edge_rbe2 = edge_rbe2,
                                            edge_rbe3 = edge_rbe3,

                                            stress = stress,
                                            tetra_edge_id = tetra_edge_id,
                                            tetra_edge_orientation = tetra_edge_orientation,
                                            coords = coords,
                                            elements = elements,
                                            edge = edge,
                                            displacement = displacement,
                                            element_stress_tensor = element_stress_tensor,
                                            element_vm_stress = element_vm_stress,
                                            E = self.E,
                                            nu = self.nu,
                                            rho = self.rho))


        print('# of train dataset:', len(train_graph_list))
        print('# of valid dataset', len(valid_graph_list))

        self.input_dim = x.shape[1]
        self.bc_dim = bc.shape[1]
        self.output_dim = y.shape[1]

        return train_graph_list, valid_graph_list
   
    def make_model_components(self, cfg: DictConfig):

        self.model = baseline(
             input_dim = self.input_dim,
             bc_dim = self.bc_dim,
             output_dim = self.output_dim,
             hidden_dim = cfg.arch.processor.hidden_dim,
             n_layers_enc = cfg.arch.encoder.n_layers_enc,
             n_layers_pro = cfg.arch.processor.n_layers_pro,
             n_layers_dec = cfg.arch.decoder.n_layers_dec,
         ).to(self.device)

        summary(self.model)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.scheduler.initial_lr, weight_decay = cfg.scheduler.weight_decay)


    def train(self, cfg):
        for epoch in range(self.max_epochs):
            train_loss = 0
            self.model.train()
            
            for graph in self.train_loader:
                self.optimizer.zero_grad()
                pred = self.model(graph.x.to(self.device), graph.bc.to(self.device), graph.edge_index.to(self.device), graph.edge_weight.to(self.device))
                y = graph.element_stress_tensor.to(self.device)

                pred[graph.edge_rbe2] = torch.tensor([0,0,0], dtype=torch.float32).to(self.device)

                stress_tensor_calculated = compute_element_stress_by_displacement_gradient(graph.coords.to(self.device), graph.elements.to(self.device), pred, graph.tetra_edge_id.to(self.device), graph.tetra_edge_orientation.to(self.device), graph.E.to(self.device), graph.nu.to(self.device), self.device)[0] / 1e6
                loss = self.criterion(stress_tensor_calculated.flatten(), y.flatten())

                # loss = self.criterion(pred.flatten(), y.flatten())
                # if epoch > 150:
                #     loss += 0.0001 * torch.norm(compute_gradient_field_loop_conservation_loss(graph.x, graph.tetra_edge_id, graph.tetra_edge_orientation, device=self.device).flatten())
                #     stress_calculated = compute_element_stress_by_displacement_gradient(graph.coords, graph.elements, pred, graph.tetra_edge_id, graph.tetra_edge_orientation, graph.E, graph.nu, self.device)[1]
                #     loss += self.criterion(stress_calculated / 1e6, graph.element_vm_stress) / 1e10
                
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                graph.to("cpu").detach()
                
            train_loss /= len(self.train_loader)
            self.train_save_ckpt(train_loss, epoch)
            valid_loss = self.val(epoch)
            self.valid_save_ckpt(valid_loss, epoch)
            
            lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1:>4}/{self.max_epochs:>4}] | Train Loss: {train_loss:.6f} | Val Loss: {valid_loss:.6f} | LR: {lr:.0e}")
            
            if cfg.experiment.wandb:
                wandb.log({"train_loss": train_loss, "valid_loss": valid_loss})

        print(f"Best Train Loss: {self.min_train_loss:.6f} at epoch {self.min_train_epoch+1}")
        print(f"Best Valid Loss: {self.min_val_loss:.6f} at epoch {self.min_val_epoch+1}")

    @torch.no_grad()
    def val(self, epoch):
        valid_loss = 0
        self.model.eval()
        for graph in self.valid_loader:
            pred = self.model(graph.x.to(self.device), graph.bc.to(self.device), graph.edge_index.to(self.device), graph.edge_weight.to(self.device))
            y = graph.element_stress_tensor.to(self.device)

            pred[graph.edge_rbe2] = torch.tensor([0,0,0], dtype=torch.float32).to(self.device)

            stress_tensor_calculated = compute_element_stress_by_displacement_gradient(graph.coords.to(self.device), graph.elements.to(self.device), pred, graph.tetra_edge_id.to(self.device), graph.tetra_edge_orientation.to(self.device), graph.E.to(self.device), graph.nu.to(self.device), self.device)[0] / 1e6
            loss = self.criterion(stress_tensor_calculated.flatten(), y.flatten())

            # loss = self.criterion(pred.flatten(), y.flatten())
            # if epoch > 150:
            #     loss += 0.0001 * torch.norm(compute_gradient_field_loop_conservation_loss(graph.x, graph.tetra_edge_id, graph.tetra_edge_orientation, device=self.device).flatten())
            #     stress_calculated = compute_element_stress_by_displacement_gradient(graph.coords, graph.elements, pred, graph.tetra_edge_id, graph.tetra_edge_orientation, graph.E, graph.nu, self.device)[1]
            #     loss += self.criterion(stress_calculated / 1e6, graph.element_vm_stress) / 1e10
            
            valid_loss += loss.item()

        valid_loss /= len(self.valid_loader)
        return valid_loss


    def train_save_ckpt(self, loss, epoch):
        if loss < self.min_train_loss:
            self.min_train_loss = loss
            self.min_train_epoch = epoch
            torch.save(self.model.state_dict(), os.path.join(self.ckpt_path, "st_best_train_loss_2.pt"))


    def valid_save_ckpt(self, loss, epoch):
        if loss < self.min_val_loss:
            self.min_val_loss = loss
            self.min_val_epoch = epoch            
            torch.save(self.model.state_dict(), os.path.join(self.ckpt_path, "st_best_valid_loss_2.pt"))


@hydra.main(config_path="config", version_base="1.1", config_name="config.yaml") 

def main(cfg: DictConfig):
    ###### 시드 고정 ######
    seed = cfg.experiment.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cfg.experiment.device = "cuda:2"

    cfg.experiment.wandb = True
    cfg.experiment.wandb_api_key = "c55bae1710d00a1d3b5e6852e0fb7fdeafa5de57"
    cfg.experiment.wandb_project_name = "SimJEB Laplacian"

    cfg.arch.encoder.n_layers_enc = 2
    cfg.arch.processor.hidden_dim = 64
    cfg.arch.processor.n_layers_pro = 4
    cfg.arch.decoder.n_layers_dec = 2

    cfg.dataset.max_epochs = 3000

    ###### weight and bias 설정 ######
    if cfg.experiment.wandb:
        os.environ['WANDB_API_KEY'] = cfg.experiment.wandb_api_key
        wandb.init(project=cfg.experiment.wandb_project_name, name="GATv2 (st)")

    for key in list(cfg.keys()):
        print(f"{key}: {cfg[key]}")

    ###### 학습 및 평가 시작 ######
    trainer = Trainer(cfg)
    trainer.train(cfg)

if __name__  == "__main__":
    main()