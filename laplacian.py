import torch

import sys
import importlib.util

solver_dir = "/home/ubuntu/SML/solver"

sys.path.append(solver_dir)

file_path = "/home/ubuntu/SML/solver/solver.py"
spec = importlib.util.spec_from_file_location("solver", file_path)
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)

def tetrahedral_to_full_edge_data(elements, device='cuda:0'):
    """
    Given a tetrahedral mesh 'elements' of shape [M,4] (each row is [n1,n2,n3,n4]), compute:

      - unique_edges: tensor of shape [E,2] containing unique global edge node indices,
                      stored in canonical order (sorted order).
      - tet_edge_ids: tensor of shape [M,6] where each row contains the global edge index for
                      the six edges of that tetrahedron (order: [n1,n2], [n1,n3], [n1,n4],
                      [n2,n3], [n3,n4], [n4,n1]).
      - tet_orientations: tensor of shape [M,6] with +1 or -1 indicating whether the
                      tetrahedron’s edge direction agrees with the canonical order.
    """
    M = elements.shape[0]
    
    n1 = elements[:, 0]
    n2 = elements[:, 1]
    n3 = elements[:, 2]
    n4 = elements[:, 3]
    
    e1 = torch.stack([n1, n2], dim=1)  # [n1, n2]
    e2 = torch.stack([n1, n3], dim=1)  # [n1, n3]
    e3 = torch.stack([n1, n4], dim=1)  # [n1, n4]
    e4 = torch.stack([n2, n3], dim=1)  # [n2, n3]
    e5 = torch.stack([n3, n4], dim=1)  # [n3, n4]
    e6 = torch.stack([n4, n2], dim=1)  # [n4, n2]
    
    tet_edges = torch.stack([e1, e2, e3, e4, e5, e6], dim=1)  
    
    tet_edges_flat = tet_edges.view(-1, 2)
    
    tet_edges_canonical, _ = torch.sort(tet_edges_flat, dim=1)
    
    unique_edges, inv_indices = torch.unique(tet_edges_canonical, return_inverse=True, dim=0)
    
    tet_edge_ids = inv_indices.view(M, 6)
    
    orientations_flat = torch.where(
        (tet_edges_flat[:, 0] == tet_edges_canonical[:, 0]) & 
        (tet_edges_flat[:, 1] == tet_edges_canonical[:, 1]),
        torch.tensor(1, device=device),
        torch.tensor(-1, device=device)
    )
    
    tet_orientations = orientations_flat.view(M, 6)
    
    return unique_edges, tet_edge_ids, tet_orientations # [E,2], [M,6], [M,6]

def compute_edge_edge(tetra_edge_id, device="cuda:0"):
    edge_edge = tetra_edge_id[:, [[0, 1], [0, 2], [0, 3], [0, 5], [1, 2], [1, 3], [1, 4], [2, 4], [2, 5], [3, 4], [3, 5], [4, 5]]]
    edge_edge = edge_edge.view(-1, 2)  # [M, 2]
    edge_edge = torch.unique(edge_edge, dim=0)  # [E, 2]

    return edge_edge

def compute_target_gradient(coords, edge, device="cuda:0"):
    n1 = edge[:, 0]
    n2 = edge[:, 1]
    
    coord1 = coords[n1]  # shape [E, 3]
    coord2 = coords[n2]  # shape [E, 3]
    
    target_gradient = coord2 - coord1  # shape [E, 3]
    
    return target_gradient

def compute_element_stress_by_displacement_gradient(coords, elements, displacement_gradient, tetra_edge_id, tetra_orientation, E, nu, device="cuda:0", dtype=torch.float32):
    M = elements.shape[0]
    coords = coords.to(device=device, dtype=dtype)
    elements = elements.to(device)
    displacement_gradient = displacement_gradient.to(device=device, dtype=dtype)

    C = solver.compute_elasticity_matrix(E, nu, device, dtype)  # [6,6]

    disp_elem = displacement_gradient[tetra_edge_id][:,[0,1,2],:] * tetra_orientation[:,[0,1,2]].unsqueeze(-1) # [M, 3, 3]
    disp_elem = torch.cat((torch.zeros(M, 1, 3, device=device, dtype=dtype), disp_elem), dim=1) # [M, 4, 3]
    disp_elem = disp_elem.reshape(M, -1) # [M, 12]

    B = solver.compute_c3d4_B_matrix(coords, elements, device, dtype)  # [M,6,12]
    strain = torch.bmm(B, disp_elem.unsqueeze(2)).squeeze(2)  # [M,6]
    stress = torch.matmul(strain, C.t())  # [M,6]
    stress_tensor = solver.compute_stress_tensor(stress)  # [M,3,3]
    element_vm_stress = solver.compute_von_mises_stress(stress_tensor)  # [M]

    return stress_tensor, element_vm_stress # [M,3,3], [M]

def compute_gradient_field_loop_conservation_loss(gradient, tetra_edge_id, tetra_edge_orientation, device="cuda:0", dtype=torch.float32):
    n1_n2_n3 = (gradient[tetra_edge_id][:,[0,3],:] * tetra_edge_orientation[:,[0,3]].unsqueeze(-1)).sum(dim=1)
    n1_n3 = (gradient[tetra_edge_id][:,[1],:] * tetra_edge_orientation[:,[1]].unsqueeze(-1)).squeeze(1)

    n1_n4_n2 = (gradient[tetra_edge_id][:,[2,5],:] * tetra_edge_orientation[:,[2,5]].unsqueeze(-1)).sum(dim=1)
    n1_n2 = (gradient[tetra_edge_id][:,[0],:] * tetra_edge_orientation[:,[0]].unsqueeze(-1)).squeeze(1)

    n1_n3_n4 = (gradient[tetra_edge_id][:,[1,4],:] * tetra_edge_orientation[:,[1,4]].unsqueeze(-1)).sum(dim=1)
    n1_n4 = (gradient[tetra_edge_id][:,[2],:] * tetra_edge_orientation[:,[2]].unsqueeze(-1)).squeeze(1)

    n2_n3_n4 = (gradient[tetra_edge_id][:,[3,4],:] * tetra_edge_orientation[:,[3,4]].unsqueeze(-1)).sum(dim=1)
    n4_n2 = (gradient[tetra_edge_id][:,[5],:] * tetra_edge_orientation[:,[5]].unsqueeze(-1)).squeeze(1)

    loss = torch.cat((n1_n2_n3 - n1_n3, n1_n4_n2 - n1_n2, n1_n3_n4 - n1_n4, n2_n3_n4 + n4_n2), dim=0)

    return loss 


def recover_nodal_field_spanning_tree(edge, gradient, num_nodes, start_nodes, device="cuda:0"):
    edge = edge.to(device)
    gradient = gradient.to(device)
    num_nodes = int(num_nodes)
    start_nodes = start_nodes.to(device)
    src = edge[:, 0]
    dst = edge[:, 1]
    src_all = torch.cat([src, dst])
    dst_all = torch.cat([dst, src])
    grad_all = torch.cat([gradient, -gradient], dim=0)
    f = torch.zeros((num_nodes, 3), device=device, dtype=gradient.dtype)
    visited = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    current = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    current[start_nodes] = True
    visited[start_nodes] = True

    while current.any():
        mask = current[src_all] & (~visited[dst_all])
        if not mask.any():
            break
        cand_nodes = dst_all[mask]
        cand_parents = src_all[mask]
        cand_values = f[cand_parents] + grad_all[mask]
        unique_nodes, inv_indices = torch.unique(cand_nodes, sorted=True, return_inverse=True)
        sums = torch.zeros((unique_nodes.shape[0], 3), device=device, dtype=cand_values.dtype)
        counts = torch.zeros(unique_nodes.shape[0], device=device, dtype=cand_values.dtype)
        sums = sums.index_add(0, inv_indices, cand_values)
        counts = counts.index_add(0, inv_indices, torch.ones_like(cand_nodes, dtype=cand_values.dtype))
        new_values = sums / counts.unsqueeze(1)
        f[unique_nodes] = new_values
        visited[unique_nodes] = True
        current = torch.zeros_like(current)
        current[unique_nodes] = True
        
    return f

def compute_hop_iteration_for_edge(edge_edge, num_edge, start_edges, device="cuda:0"):
    edge_edge = edge_edge.to(device)
    start_edges = start_edges.to(device)
    src = edge_edge[:, 0]
    dst = edge_edge[:, 1]
    src_all = torch.cat([src, dst])
    dst_all = torch.cat([dst, src])
    ones = torch.ones(edge_edge.shape[0], device=device)
    weight_all = torch.cat([ones, ones])
    f = torch.full((int(num_edge),), float('inf'), device=device)
    f[start_edges] = 0.0
    current = torch.zeros(int(num_edge), dtype=torch.bool, device=device)
    current[start_edges] = True
    improved = True
    while improved:
        mask = current[src_all]
        if not mask.any():
            break
        cand_src = src_all[mask]
        cand_dst = dst_all[mask]
        cand_values = f[cand_src] + weight_all[mask]
        unique_nodes, inv_indices = torch.unique(cand_dst, sorted=True, return_inverse=True)
        new_values = torch.full((unique_nodes.shape[0],), float('inf'), device=device)
        new_values = new_values.scatter_reduce(0, inv_indices, cand_values, reduce="amin", include_self=False)
        old_values = f[unique_nodes]
        update_mask = new_values < old_values
        if update_mask.any():
            f[unique_nodes[update_mask]] = new_values[update_mask]
            new_current = torch.zeros_like(current)
            new_current[unique_nodes[update_mask]] = True
            current = new_current
        else:
            improved = False
    return f


def compute_length_sum_for_edge(edge_edge, num_edge, start_edges, node_weight, device="cuda:0"):
    edge_edge = edge_edge.to(device)
    node_weight = node_weight.to(device)
    start_edges = start_edges.to(device)
    src = edge_edge[:, 0]
    dst = edge_edge[:, 1]
    src_all = torch.cat([src, dst])
    dst_all = torch.cat([dst, src])
    f = torch.full((int(num_edge),), float('inf'), device=device)
    f[start_edges] = 0.0
    current = torch.zeros(int(num_edge), dtype=torch.bool, device=device)
    current[start_edges] = True
    improved = True
    while improved:
        mask = current[src_all]
        if not mask.any():
            break
        cand_src = src_all[mask]
        cand_dst = dst_all[mask]
        cand_values = f[cand_src] + node_weight[cand_dst]
        unique_nodes, inv_indices = torch.unique(cand_dst, sorted=True, return_inverse=True)
        new_values = torch.full((unique_nodes.shape[0],), float('inf'), device=device)
        new_values = new_values.scatter_reduce(0, inv_indices, cand_values, reduce="amin", include_self=False)
        old_values = f[unique_nodes]
        update_mask = new_values < old_values
        if update_mask.any():
            f[unique_nodes[update_mask]] = new_values[update_mask]
            new_current = torch.zeros_like(current)
            new_current[unique_nodes[update_mask]] = True
            current = new_current
        else:
            improved = False
    return f