import torch

import sys
import importlib.util

solver_dir = "/home/ubuntu/SML/solver"

sys.path.append(solver_dir)

file_path = "/home/ubuntu/SML/solver/solver.py"
spec = importlib.util.spec_from_file_location("solver", file_path)
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)


def shell_to_full_edge_data(triangle_elements, square_elements, device='cuda:0'):
    """
    Given a mixed mesh defined by:
      - triangle_elements: tensor of shape [T,3] (each row is [n1,n2,n3])
      - square_elements: tensor of shape [S,4] (each row is [n1,n2,n3,n4])
    
    This function computes:
      - unique_edges: tensor of shape [E,2] containing unique global edge node indices,
                      stored in canonical order (sorted order).
      - tri_edge_ids: tensor of shape [T,3] where each row contains the global edge index for
                      the three edges of the triangle (order: [n1,n2], [n2,n3], [n3,n1]).
      - tri_orientations: tensor of shape [T,3] with +1 or -1 indicating whether the
                          triangle’s edge direction agrees with the canonical order.
      - square_edge_ids: tensor of shape [S,4] where each row contains the global edge index for
                         the four edges of the square (order: [n1,n2], [n2,n3], [n3,n4], [n4,n1]).
      - square_orientations: tensor of shape [S,4] with +1 or -1 indicating whether the
                             square’s edge direction agrees with the canonical order.
    """
    T = triangle_elements.shape[0]
    t_n1 = triangle_elements[:, 0]
    t_n2 = triangle_elements[:, 1]
    t_n3 = triangle_elements[:, 2]
    
    tri_e1 = torch.stack([t_n1, t_n2], dim=1)  # edge [n1, n2]
    tri_e2 = torch.stack([t_n2, t_n3], dim=1)  # edge [n2, n3]
    tri_e3 = torch.stack([t_n3, t_n1], dim=1)  # edge [n3, n1]
    
    tri_edges = torch.stack([tri_e1, tri_e2, tri_e3], dim=1)  # shape [T, 3, 2]
    tri_edges_flat = tri_edges.view(-1, 2)  # shape [T*3, 2]
    
    S = square_elements.shape[0]
    s_n1 = square_elements[:, 0]
    s_n2 = square_elements[:, 1]
    s_n3 = square_elements[:, 2]
    s_n4 = square_elements[:, 3]
    
    square_e1 = torch.stack([s_n1, s_n2], dim=1)  # edge [n1, n2]
    square_e2 = torch.stack([s_n2, s_n3], dim=1)  # edge [n2, n3]
    square_e3 = torch.stack([s_n3, s_n4], dim=1)  # edge [n3, n4]
    square_e4 = torch.stack([s_n4, s_n1], dim=1)  # edge [n4, n1]
    
    square_edges = torch.stack([square_e1, square_e2, square_e3, square_e4], dim=1)  # shape [S, 4, 2]
    square_edges_flat = square_edges.view(-1, 2)  # shape [S*4, 2]
    
    all_edges_flat = torch.cat([tri_edges_flat, square_edges_flat], dim=0)  # shape [(T*3 + S*4), 2]
    
    all_edges_canonical, _ = torch.sort(all_edges_flat, dim=1)
    
    unique_edges, inv_indices = torch.unique(all_edges_canonical, return_inverse=True, dim=0)
    
    tri_inv = inv_indices[:T*3].view(T, 3)
    square_inv = inv_indices[T*3:].view(S, 4)
    
    orientations_all = torch.where(
        (all_edges_flat[:, 0] == all_edges_canonical[:, 0]) &
        (all_edges_flat[:, 1] == all_edges_canonical[:, 1]),
        torch.tensor(1, device=device),
        torch.tensor(-1, device=device)
    )
    
    tri_orientations = orientations_all[:T*3].view(T, 3)
    square_orientations = orientations_all[T*3:].view(S, 4)
    
    return unique_edges, tri_inv, tri_orientations, square_inv, square_orientations


def compute_edge_edge_shell(tri_edge_id, square_edge_id):
    tri_pairs = tri_edge_id[:, [[0, 1], [0, 2], [1, 2]]]
    tri_pairs = tri_pairs.view(-1, 2)
    
    square_pairs = square_edge_id[:, [[0, 1], [1, 2], [2, 3], [3, 0]]]
    square_pairs = square_pairs.view(-1, 2)

    all_pairs = torch.cat([tri_pairs, square_pairs], dim=0)
    
    edge_edge = torch.unique(all_pairs, dim=0)
    
    return edge_edge 


def compute_gradient_field_loop_conservation_loss_shell(gradient, tri_edge_id, tri_edge_orientation, square_edge_id, square_edge_orientation):
    tri_loss = (gradient[tri_edge_id] * tri_edge_orientation.unsqueeze(-1)).sum(dim=1)  # shape [T, D]

    square_loss = (gradient[square_edge_id] * square_edge_orientation.unsqueeze(-1)).sum(dim=1)  # shape [S, D]
    
    loss = torch.cat([tri_loss, square_loss], dim=0)
    
    return loss


