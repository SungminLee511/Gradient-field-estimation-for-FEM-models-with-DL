
##### 모든 인풋은 batch화한 torch.Tensor
##### cuda 사용시 끝나면 cpu로 다시 가져오는 부분이 없기 때문에 따로 가져오기
##### N은 노드 개수, E는 edge 개수, M은 element 개수, K는 겉면 개수, S는 겉면제외 면 개수
##### 2*S + K = 4*M
##### 모든 기본값은 cuda:0, torch.float32



import torch
import numpy as np
import plotly.graph_objects as go
import pyvista as pv





#################################################################################################
#########################################   기본 함수   ###########################################
#################################################################################################

def human_readable_number(num):
    if abs(num) >= 1_000_000_000_000_000_000:
        return f"{num / 1_000_000_000_000_000_000:.1f}Quint"
    elif abs(num) >= 1_000_000_000_000_000:
        return f"{num / 1_000_000_000_000_000:.1f}Quad"
    elif abs(num) >= 1_000_000_000_000:
        return f"{num / 1_000_000_000_000:.1f}T"
    elif abs(num) >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f}B"
    elif abs(num) >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif abs(num) >= 1_000:
        return f"{num / 1_000:.1f}K" 
    else:
        return f"{num:.1f}"

def vtk_loader_to_torch(file_path, element_type, device="cuda:0", dtype=torch.float32):
    """
    Load a VTK file and convert it to a torch.Tensor.

    Args:
        file_path (str): Path to the VTK file.
        element_type (str): Element type (e.g. "c3d4")
        device (str): Device to store the data (default: "cuda:0")
        dtype (torch.dtype): Data type (default: torch.float32)
    
    Returns:
        points (torch.Tensor): Node coordinates [N, 3]
        tetrahedrons (torch.Tensor): Tetrahedral element connectivity [M, 4] etc.
    """
    mesh = pv.read(file_path)
    
    points = torch.tensor(mesh.points, device=device, dtype=dtype)  

    if element_type == "c3d4":
        cells = mesh.cells.reshape(-1, 5)[:, 1:]  
        element = torch.tensor(cells, device=device, dtype=torch.long)  
    elif element_type == "c3d10":
        cells = mesh.cells.reshape(-1, 11)[:, 1:]  
        element = torch.tensor(cells, device=device, dtype=torch.long)
    elif element_type == "c3d8":
        cells = mesh.cells.reshape(-1, 9)[:, 1:]  
        element = torch.tensor(cells, device=device, dtype=torch.long)
    elif element_type == "c3d20":
        cells = mesh.cells.reshape(-1, 21)[:, 1:]  
        element = torch.tensor(cells, device=device, dtype=torch.long)
    elif element_type == "c3d6":
        cells = mesh.cells.reshape(-1, 7)[:, 1:]  
        element = torch.tensor(cells, device=device, dtype=torch.long)
    elif element_type == "c3d15":
        cells = mesh.cells.reshape(-1, 16)[:, 1:]  
        element = torch.tensor(cells, device=device, dtype=torch.long)
    elif element_type == "s3":
        cells = mesh.cells.reshape(-1, 4)[:, 1:]  
        element = torch.tensor(cells, device=device, dtype=torch.long)
    elif element_type == "s6":
        cells = mesh.cells.reshape(-1, 7)[:, 1:]  
        element = torch.tensor(cells, device=device, dtype=torch.long)
    elif element_type == "s4":
        cells = mesh.cells.reshape(-1, 5)[:, 1:]  
        element = torch.tensor(cells, device=device, dtype=torch.long)
    elif element_type == "s8":
        cells = mesh.cells.reshape(-1, 9)[:, 1:]  
        element = torch.tensor(cells, device=device, dtype=torch.long)
    else:
        raise ValueError("Invalid element type.")

    return points, element

##### 시각화 관련
def visualize_mesh(coords, elements, element_type, title='Mesh Visualization'):
    """
    Visualize a 3D mesh using Plotly go.Mesh3d.
    
    Parameters:
      coords (torch.Tensor or array-like): Node coordinates [N, 3].
      elements (torch.Tensor or array-like): Element connectivity.
         For volume elements the connectivity is assumed to be:
           - c3d6: shape (M, 6) wedge
           - c3d8: shape (M, 8) hexahedron
           - c3d4: shape (M, 4) tetrahedron
         For shell elements:
           - s3: shape (M, 3) (triangles)
           - s4: shape (M, 4) (quadrilaterals)
      element_type (str): One of 'c3d6', 'c3d8', 'c3d4', 's3', or 's4'.
      title (str): Title for the Plotly figure.
    """
    coords_np = coords.cpu().numpy() if isinstance(coords, torch.Tensor) else np.array(coords)
    elements_np = elements.cpu().numpy() if isinstance(elements, torch.Tensor) else np.array(elements)
    
    x = coords_np[:, 0]
    y = coords_np[:, 1]
    z = coords_np[:, 2]
    
    if element_type in ['s3', 's4']:
        if element_type == 's3':
            triangles = elements_np
        elif element_type == 's4':
            tris1 = elements_np[:, [0, 1, 2]]
            tris2 = elements_np[:, [0, 2, 3]]
            triangles = np.concatenate([tris1, tris2], axis=0)
    
    elif element_type in ['c3d6', 'c3d8', 'c3d4']:
        if element_type == 'c3d4':
            face_defs = np.array([
                [0, 1, 2],
                [0, 1, 3],
                [0, 2, 3],
                [1, 2, 3]
            ]) 
            all_faces = np.take(elements_np, face_defs, axis=1).reshape(-1, 3)
            
        elif element_type == 'c3d8':
            face_defs = np.array([
                [0, 1, 2, 3],  # bottom
                [4, 5, 6, 7],  # top
                [0, 1, 5, 4],  # side 1
                [1, 2, 6, 5],  # side 2
                [2, 3, 7, 6],  # side 3
                [3, 0, 4, 7]   # side 4
            ]) 
            all_faces = np.take(elements_np, face_defs, axis=1).reshape(-1, 4)
            tris1 = all_faces[:, [0, 1, 2]]
            tris2 = all_faces[:, [0, 2, 3]]
            all_faces = np.concatenate([tris1, tris2], axis=0) 
            
        elif element_type == 'c3d6':
            tri_bottom = elements_np[:, [0, 1, 2]]
            tri_top    = elements_np[:, [3, 4, 5]]
            quad1 = elements_np[:, [0, 1, 4, 3]]
            quad2 = elements_np[:, [1, 2, 5, 4]]
            quad3 = elements_np[:, [2, 0, 3, 5]]
            quad1_tri1 = quad1[:, [0, 1, 2]]
            quad1_tri2 = quad1[:, [0, 2, 3]]
            quad2_tri1 = quad2[:, [0, 1, 2]]
            quad2_tri2 = quad2[:, [0, 2, 3]]
            quad3_tri1 = quad3[:, [0, 1, 2]]
            quad3_tri2 = quad3[:, [0, 2, 3]]
            all_faces = np.concatenate([
                tri_bottom, tri_top,
                quad1_tri1, quad1_tri2,
                quad2_tri1, quad2_tri2,
                quad3_tri1, quad3_tri2
            ], axis=0)
        
        all_faces_sorted = np.sort(all_faces, axis=1)
        unique_faces, counts = np.unique(all_faces_sorted, axis=0, return_counts=True)
        boundary_faces = unique_faces[counts == 1]
        triangles = boundary_faces

    else:
        raise ValueError(f"Unsupported element type: {element_type}")
    
    i = triangles[:, 0].astype(np.int64)
    j = triangles[:, 1].astype(np.int64)
    k = triangles[:, 2].astype(np.int64)
    
    mesh = go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        opacity=0.5,
        color='lightgrey',
        flatshading=True,
    )
    
    fig = go.Figure(data=[mesh])
    fig.update_layout(
        title=title,
        scene=dict(aspectmode='data')
    )
    fig.show()

def visualize_node_with_value(coords, value, size=0.5, colorscale='Viridis', title='Node-wise Von Mises Stress Visualization', bar_title='Von Mises Stress'):
    """
    노드 단위 시각화

    Input:
        coords (torch.Tensor): Node coordinates [N, 3]
        value_capped (torch.Tensor): Capped node-wise von Mises stress [N]
    """
    x = coords[:, 0].cpu().numpy()
    y = coords[:, 1].cpu().numpy()
    z = coords[:, 2].cpu().numpy()
    stress = value.cpu().numpy()

    hover_text = [f"Node {i}<br>Stress: {s:.2f}" for i, s in enumerate(stress)]

    fig = go.Figure(data=go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=size,
            color=stress,
            colorscale=colorscale,
            colorbar=dict(title=bar_title),
            opacity=0.8
        ),
        text=hover_text,
        hoverinfo='text'
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        title={
            'text': title,
            'x': 0.5, 
            'xanchor': 'center'
        }
    )

    fig.show()

def visualize_target_nodes(coords, node_ids=None, marker_size=0.2, target_marker_size=2.0):
    """
    Plot nodes in 3D with plotly.
    
    coords: [N, 3] tensor of node coordinates
    node_ids: [M] tensor of node indices to be colored red
    marker_size: size of the markers in the plot
    """
    N = coords.shape[0]
    colors = ['blue'] * N
    marker_size = [marker_size] * N
    if node_ids is not None:
        for node_id in node_ids:
            colors[node_id] = 'red'
            marker_size[node_id] = target_marker_size
    
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=coords[:, 0].cpu().numpy(),
        y=coords[:, 1].cpu().numpy(),
        z=coords[:, 2].cpu().numpy(),
        mode='markers',
        marker=dict(
            size=marker_size,
            color=colors
        )
    ))
    fig.update_layout(
        scene=dict(
            aspectmode='data'
        ),
        title="3D Node Visualization"
    )
    fig.show()


#################################################################################################
##########################################   통합 함수   ##########################################
#################################################################################################

def compute_elasticity_matrix(E, nu, device="cuda:0", dtype=torch.float32):
    """
    탄성계수 행렬 D를 만드는 함수
    Element의 형태와 관련 없이 항상 6*6 행렬이다.

    stress = D @ strain
    stress = [S11, S22, S13, S23, S13, S12]
    
    Input:
        E (float): Young's modulus
        nu (float): Poisson's ratio
    
    Output:
        D (torch.Tensor): 6*6 D matrix.
    """
    coef = E / ((1 + nu) * (1 - 2 * nu))
    D = coef * torch.tensor([
        [1 - nu,     nu,     nu,       0,       0,       0],
        [    nu, 1 - nu,     nu,       0,       0,       0],
        [    nu,     nu, 1 - nu,       0,       0,       0],
        [     0,      0,      0, (1 - 2 * nu) / 2, 0,       0],
        [     0,      0,      0,       0, (1 - 2 * nu) / 2, 0],
        [     0,      0,      0,       0,       0, (1 - 2 * nu) / 2]
    ], device=device, dtype=dtype)
    return D

def compute_stress_tensor(stress_vector):
    """
    Voigt Notation의 스트레스를 Stress Tensor로 바꿔주는 함수
    [S11, S22, S33, S23, S13, S12] -> S

    Input:
        stress_vector (torch.Tensor): Stress vector [M,6]

    Output:
        torch.Tensor: Stress tensor [M,3,3]
    """
    M = stress_vector.shape[0]
    stress_tensor = torch.zeros((M, 3, 3), device=stress_vector.device, dtype=stress_vector.dtype)
    stress_tensor[:, 0, 0] = stress_vector[:, 0]
    stress_tensor[:, 1, 1] = stress_vector[:, 1]
    stress_tensor[:, 2, 2] = stress_vector[:, 2]
    stress_tensor[:, 0, 1] = stress_vector[:, 3]
    stress_tensor[:, 1, 0] = stress_vector[:, 3]
    stress_tensor[:, 0, 2] = stress_vector[:, 5]
    stress_tensor[:, 2, 0] = stress_vector[:, 5]
    stress_tensor[:, 1, 2] = stress_vector[:, 4]
    stress_tensor[:, 2, 1] = stress_vector[:, 4]
    return stress_tensor

def compute_von_mises_stress(stress_tensor):
    """
    Stress Tensor가 주어졌을때 Von Mises Stress를 반환하는 함수.
    
    Input:
        stress_tensor (torch.Tensor): Stress tensor [M,3,3].
    
    Returns:
        torch.Tensor: Von Mises stress [M].
    """
    s_xx = stress_tensor[:, 0, 0]
    s_yy = stress_tensor[:, 1, 1]
    s_zz = stress_tensor[:, 2, 2]
    s_xy = stress_tensor[:, 0, 1]
    s_xz = stress_tensor[:, 0, 2]
    s_yz = stress_tensor[:, 1, 2]
    
    von_mises = torch.sqrt(
        ((s_xx - s_yy)**2 + (s_yy - s_zz)**2 + (s_zz - s_xx)**2 +
         6*(s_xy**2 + s_yz**2 + s_xz**2)) / 2
    )
    return von_mises

def to_c3d4(elements, device="cuda:0"):
    n = elements.shape[1]
    if n == 6: 
        return c3d6_to_c3d4(elements, device)
    if n == 8:
        return c3d8_to_c3d4(elements, device)
    if n == 10:
        return c3d10_to_c3d4(elements, device)
    if n == 20:
        return c3d20_to_c3d4(elements, device)

def to_2nd_order(coords, elements, rbe2=None, rbe3=None, device="cuda:0", dtype=torch.float32):
    n = elements.shape[1]
    if n == 4:
        return c3d4_to_c3d10(coords, elements, rbe2, rbe3, dtype=dtype)

def integral_points(element_type, device="cuda:0"):
    element_type = element_type.lower()
    if element_type == "c3d8": return c3d8_integration_points(device)
    if element_type == "c3d10": return c3d10_integration_points(device)
    if element_type == "c3d20": return c3d20_integration_points(device)
    if element_type == "c3d6": return c3d6_integration_points(device)
    if element_type == "c3d15": return c3d15_integration_points(device)
    raise ValueError(f"Unsupported element type: {element_type}")

def compute_Jacobian(coords, elements, element_type, integral_point=None, device="cuda:0"):
    element_type = element_type.lower()
    if element_type == "c3d8": return compute_c3d8_Jacobian(coords, elements, integral_point, device)
    if element_type == "c3d8i": return compute_c3d8_Jacobian(coords, elements, integral_point, device)
    if element_type == "c3d10": return compute_c3d10_Jacobian(coords, elements, integral_point, device)
    if element_type == "c3d20": return compute_c3d20_Jacobian(coords, elements, integral_point, device)
    if element_type == "c3d6": return compute_c3d6_Jacobian(coords, elements, integral_point, device)
    if element_type == "c3d15": return compute_c3d15_Jacobian(coords, elements, integral_point, device)
    raise ValueError(f"Unsupported element type: {element_type}")

def compute_shape_gradients(coords, elements, element_type, integral_point=None, device="cuda:0"):
    element_type = element_type.lower()
    if element_type == "c3d8": return compute_c3d8_shape_gradients(coords, elements, integral_point, device)
    if element_type == "c3d10": return compute_c3d10_shape_gradients(coords, elements, integral_point, device)
    if element_type == "c3d20": return compute_c3d20_shape_gradients(coords, elements, integral_point, device)
    if element_type == "c3d6": return compute_c3d6_shape_gradients(coords, elements, integral_point, device)
    if element_type == "c3d15": return compute_c3d15_shape_gradients(coords, elements, integral_point, device)
    raise ValueError(f"Unsupported element type: {element_type}")

def compute_B_matrix(coords, elements, integral_point, element_type, device="cuda:0", dtype=torch.float32):
    element_type = element_type.lower()
    if element_type == "c3d4": return compute_c3d4_B_matrix(coords, elements, device, dtype)
    if element_type == "c3d8": return compute_c3d8_B_matrix(coords, elements, integral_point, device)
    if element_type == "c3d10": return compute_c3d10_B_matrix(coords, elements, integral_point, device)
    if element_type == "c3d20": return compute_c3d20_B_matrix(coords, elements, integral_point, device)
    if element_type == "c3d6": return compute_c3d6_B_matrix(coords, elements, integral_point, device)
    if element_type == "c3d15": return compute_c3d15_B_matrix(coords, elements, integral_point, device)
    raise ValueError(f"Unsupported element type: {element_type}")

def compute_element_stress(coords, elements, displacement, E, nu, element_type, integral_point=None, single=True, device="cuda:0", dtype=torch.float32):
    element_type = element_type.lower()
    if element_type == "c3d4": return compute_c3d4_element_stress(coords, elements, displacement, E, nu, device, dtype)
    if element_type == "c3d8": return compute_c3d8_element_stress(coords, elements, displacement, E, nu, integral_point, single, device, dtype)
    if element_type == "c3d10": return compute_c3d10_element_stress(coords, elements, displacement, E, nu, integral_point, single, device, dtype)
    if element_type == "c3d20": return compute_c3d20_element_stress(coords, elements, displacement, E, nu, integral_point, single, device, dtype)
    if element_type == "c3d6": return compute_c3d6_element_stress(coords, elements, displacement, E, nu, integral_point, single, device, dtype)
    if element_type == "c3d15": return compute_c3d15_element_stress(coords, elements, displacement, E, nu, integral_point, single, device, dtype)
    raise ValueError(f"Unsupported element type: {element_type}")

def compute_K_matrix(coords, elements, element_type, E, nu, integral_point=None, single=True, device="cuda:0", dtype=torch.float32):
    element_type = element_type.lower()
    if element_type == "c3d4": return compute_c3d4_K_matrix(coords, elements, E, nu, device, dtype)
    if element_type == "c3d8": return compute_c3d8_K_matrix(coords, elements, E, nu, integral_point, single, device, dtype)
    if element_type == "c3d10": return compute_c3d10_K_matrix(coords, elements, E, nu, integral_point, single, device, dtype)
    if element_type == "c3d20": return compute_c3d20_K_matrix(coords, elements, E, nu, integral_point, single, device, dtype)
    if element_type == "c3d6": return compute_c3d6_K_matrix(coords, elements, E, nu, integral_point, single, device, dtype)
    if element_type == "c3d15": return compute_c3d15_K_matrix(coords, elements, E, nu, integral_point, single, device, dtype)
    raise ValueError(f"Unsupported element type: {element_type}")

def compute_nodal_forces(K, elements, displacement, device="cuda:0", dtype=torch.float32):
    """
    Computes global nodal forces for various element types without assembling the global stiffness matrix.

    Args:
        K (torch.Tensor): Local stiffness matrices [M, DOF, DOF]
        elements (torch.Tensor): Element connectivity [M, num_nodes]
        displacement (torch.Tensor): Nodal displacements [N, 3]
        num_nodes (int): Number of nodes per element (e.g., 8, 10, 20)
        device (str, optional): Device to perform computations on. Defaults to "cuda:0".
        dtype (torch.dtype, optional): Data type of the tensor. Defaults to torch.float32.

    Returns:
        torch.Tensor: Global nodal force vector [N, 3]
    """
    K = K.to(device=device, dtype=dtype)
    elements = elements.to(device=device, dtype=torch.long)
    displacement = displacement.to(device=device, dtype=dtype)

    M = K.shape[0]  
    N = displacement.shape[0] 

    dofs = elements.unsqueeze(-1) * 3 + torch.tensor([0, 1, 2], device=device).view(1, 1, 3)
    dofs = dofs.view(M, -1)  # [M, dof_per_element]

    u_local = displacement.reshape(-1)[dofs]  # [M, dof_per_element]

    F_local = torch.bmm(K, u_local.unsqueeze(-1)).squeeze(-1)  # [M, dof]

    F_global = torch.zeros((N * 3,), device=device, dtype=dtype)  # [N*3]

    F_global = F_global.index_add(0, dofs.view(-1), F_local.view(-1))  # [N*3]

    F_global = F_global.view(N, 3)  # [N, 3]

    return F_global 

def compute_node_vm_stress(coords, elements, element_vm_stress, device="cuda:0", dtype=torch.float32):
    """
    Assigns element-wise von Mises stress values to nodes by averaging the stresses
    from all connected elements for any given element type.

    Args:
        coords (torch.Tensor): Node coordinates [N, 3]
        elements (torch.Tensor): Element connectivity [M, num_nodes_per_element]
        element_vm_stress (torch.Tensor): Element-wise von Mises stress [M]
        device (str, optional): Device to perform computations on. Defaults to "cuda:0".
        dtype (torch.dtype, optional): Data type of the tensor. Defaults to torch.float32.

    Returns:
        torch.Tensor: Node-wise von Mises stress [N]
    """
    elements = elements.to(device)
    element_vm_stress = element_vm_stress.to(device)
    num_nodes_per_element = elements.shape[1]

    N = coords.shape[0]  

    node_indices = elements.view(-1)  

    stress_values = element_vm_stress.repeat_interleave(num_nodes_per_element)  # [M * num_nodes_per_element]

    node_stress_sum = torch.zeros(N, device=device, dtype=dtype)
    node_stress_count = torch.zeros(N, device=device, dtype=dtype)

    node_stress_sum = node_stress_sum.index_add(0, node_indices, stress_values)

    node_stress_count = node_stress_count.index_add(0, node_indices, torch.ones_like(stress_values, dtype=dtype))

    node_vm_stress = torch.where(
        node_stress_count > 0,
        node_stress_sum / node_stress_count,
        torch.zeros_like(node_stress_sum)
    )

    return node_vm_stress



#################################################################################################
########################################   Tetrahedral   ########################################
#################################################################################################


##### 기초연산 (c3d4, c3d10 모두 사용)
def compute_tetrahedral_volumes(coords, elements, device="cuda:0", dtype=torch.float32):
    """
    사면체 부피 구하기

    Input:
        coords (torch.Tensor): Node coordinates [N, 3]
        elements (torch.Tensor): Element connectivity [M, 4] 
        
    Output:
        Volume (torch.Tensor): Tetrahedral Volume [M]
    """
    coords = coords.to(device=device, dtype=dtype)
    elements = elements.to(device=device)
    
    p1 = coords[elements[:, 0]]  
    p2 = coords[elements[:, 1]] 
    p3 = coords[elements[:, 2]] 
    p4 = coords[elements[:, 3]]  

    v1 = p2 - p1
    v2 = p3 - p1 
    v3 = p4 - p1 

    det = torch.abs(torch.det(torch.stack([v1, v2, v3], dim=1)))  # [M]

    volumes = det / 6.0  # [M]

    return volumes

def compute_tetrahedral_surface_faces_with_fourth_node(elements, device="cuda:0"):
    """
    사면체 정보가 주어지면 1번만 등장하는 면(겉면)을 구하고, 이러한 겉면 사면체의 겉면이 아닌 4번째 노드 번호 또한 구하는 함수
    4번째 노드 정보는 법선을 정렬할때 사용된다.

    Input:
        elements (torch.Tensor): Element connectivity [M, 4] 
    
    Returns:
        surfaces (torch.Tensor): surface connectivity [K, 3]
        4th id (torch.Tensor): 4th node id [K]
    """
    elements = elements.to(device)

    faces = torch.cat([
        elements[:, [0, 1, 2]],  # face opposite to node 3
        elements[:, [0, 1, 3]],  # face opposite to node 2
        elements[:, [0, 2, 3]],  # face opposite to node 1
        elements[:, [1, 2, 3]],  # face opposite to node 0
    ], dim=0)
    
    fourth_nodes = torch.cat([
        elements[:, 3],  # fourth node for face [0, 1, 2]
        elements[:, 2],  # fourth node for face [0, 1, 3]
        elements[:, 1],  # fourth node for face [0, 2, 3]
        elements[:, 0],  # fourth node for face [1, 2, 3]
    ], dim=0)
    
    sorted_faces, indices = torch.sort(faces, dim=1)
    
    unique_faces, inverse_indices, counts = torch.unique(sorted_faces, dim=0, return_inverse=True, return_counts=True)
    
    surface_face_mask = counts[inverse_indices] == 1
    surface_faces = faces[surface_face_mask]
    surface_fourth_nodes = fourth_nodes[surface_face_mask]
    
    return surface_faces, surface_fourth_nodes # [K, 3], [K]

def compute_tetrahdral_surface_normals(coords, elements, device="cuda:0", dtype=torch.float32):
    """
    형상 바깥을 향하는 법선을 일괄적으로 구하는 함수
    겉면의 4번째 노드 정보가 필요하다.

    Input:
        coords (torch.Tensor): Node Coordinates [N, 3] 
        elements (torch.Tensor): Element Connectivity [M, 4]

    Output:
        surface norm (torch.Tensor): normal vector of surface faces [K, 3]
    """
    coords = coords.to(device=device, dtype=dtype)
    elements = elements.to(device=device)

    surface_faces, fourth_nodes = compute_tetrahedral_surface_faces_with_fourth_node(elements, device=device)

    p1 = coords[surface_faces[:, 0]] # [K, 3]
    p2 = coords[surface_faces[:, 1]] # [K, 3]
    p3 = coords[surface_faces[:, 2]] # [K, 3]
    
    v1 = p2 - p1 # [K, 3]
    v2 = p3 - p1 # [K, 3]
    
    normals = torch.cross(v1, v2, dim=1) # [K, 3]
    normals = normals / torch.norm(normals, dim=1, keepdim=True) # [K, 3]
    
    face_centers = (p1 + p2 + p3) / 3 # [K, 3]
    fourth_points = coords[fourth_nodes] # [K, 3]
    
    # 안쪽을 향하는 벡터 만들기
    to_fourth = fourth_points - face_centers
    to_fourth_normalized = to_fourth / torch.norm(to_fourth, dim=1, keepdim=True)
    
    # 바깥쪽을 향하는 벡터 만들기
    dot_product = (normals * to_fourth_normalized).sum(dim=1)
    normals[dot_product > 0] = -normals[dot_product > 0]
    
    return normals

def compute_tetrahedral_node_curvatures(coords, elements, device="cuda:0", dtype=torch.float32):
    """
    법선 정보와 좌표 개수를 입력받아 노드별 곡률을 구하는 함수

    Input:
        coords (torch.Tensor): Node coordinates [N, 3]
        elements (torch.Tensor): Element connectivity [M, 4]
    
    Output:
        curv (torch.Tensor): tensor of curvatures along Nodes [N, 3]
    """
    N = coords.shape[0]
    surface_faces, fourth_nodes = compute_tetrahedral_surface_faces_with_fourth_node(elements, device=device)
    surface_normals = compute_tetrahdral_surface_normals(coords, surface_faces, fourth_nodes, device=device, dtype=dtype)

    accumulated_curvatures = torch.zeros((N, 3), device=device)  # [N, 3]
    node_counts = torch.zeros(N, device=device)  # [N]

    face_normals_expanded = face_normals[:, None, :].expand(-1, 3, -1)  # [K, 3, 3]

    accumulated_curvatures.scatter_add_(0, surface_faces.view(-1, 1).expand(-1, 3), face_normals_expanded.reshape(-1, 3))

    ones = torch.ones(surface_faces.numel(), device=device)  # [3K]
    node_counts.scatter_add_(0, surface_faces.view(-1), ones)  # [N]

    node_counts[node_counts == 0] = 1

    mean_curvature_vectors = accumulated_curvatures / node_counts.view(-1, 1)  # [N, 3]

    return mean_curvature_vectors # [N, 3]

def compute_tetrahedral_normals_and_area(coords, elements, device="cuda:0", dtype=torch.float32):
    """
    각 element의 면별 법선을 구하고, element 간 면 공유 정보를 추출하는 함수
    법선의 크기는 삼각형 면이 면적이다.
    면의 순서는 아래와 같다.
    [0, 1, 2],  # Face 0
    [0, 1, 3],  # Face 1
    [1, 2, 3],  # Face 2
    [0, 2, 3]   # Face 3
    
    Inputs:
        coords (torch.Tensor): Node coordinates [N, 3]
        elements (torch.Tensor): Element connectivity [M, 4]
        
    Outputs:
        normal_vectors (torch.Tensor): Normal vectors for each face of each element [M, 4, 3]
    """
    coords = coords.to(device=device, dtype=dtype)
    elements = elements.to(device)

    M = elements.shape[0]
    normal_vectors = torch.zeros((M, 4, 3), device=device, dtype=dtype)

    coords_elem = coords[elements].to(device)  # [M, 4, 3]

    faces = torch.tensor([
        [0, 1, 2],  # Face 0
        [0, 1, 3],  # Face 1
        [1, 2, 3],  # Face 2
        [0, 2, 3]   # Face 3
    ], device=device)  # [4, 3]

    fourth_nodes = torch.stack([
        elements[:, 3],  # Face 0
        elements[:, 2],  # Face 1
        elements[:, 0],  # Face 2
        elements[:, 1]   # Face 3
    ], dim=1)  # [M, 4] 
    
    fourth_node_coords = coords[fourth_nodes]  # [M, 4, 3]
    face_coords = coords_elem[:, faces]  # [M, 4, 3, 3]

    edge1 = face_coords[:, :, 1, :] - face_coords[:, :, 0, :]  # [M, 4, 3]
    edge2 = face_coords[:, :, 2, :] - face_coords[:, :, 0, :]  # [M, 4, 3]

    normal_vectors = torch.cross(edge1, edge2, dim=2) / 2.0 # [M, 4, 3]

    centroids = face_coords.mean(dim=2)  # [M, 4, 3]
    centroid_to_fourth = fourth_node_coords - centroids  # [M, 4, 3]
    dot_products = torch.sum(normal_vectors * centroid_to_fourth, dim=2)  # [M, 4]
    flip_mask = dot_products > 0
    normal_vectors[flip_mask] = -normal_vectors[flip_mask]

    return normal_vectors # [M, 4, 3]

def identify_tetrahedral_shared_faces(elements, device="cuda:0"):
    """
    elements 정보가 주어지면 각 면이 공유되는 element 정보를 구하는 함수
    아웃풋은 [S, 2, 2]이며, 총 S개의 면에 대해 [2, 2] 연결 정보를 제공한다.
    [2, 2]:[[element id, face index],[element id, face index]]

    Inputs:
        elements (torch.Tensor): Element connectivity [M, 4] 

    Outputs:
        shared_face_indices (torch.Tensor): Indices of elements sharing each face [S, 2, 2]
    """
    M = elements.shape[0]
    elements = elements.to(device)
    
    face_node_indices = torch.tensor([
        [0, 1, 2],  # Face 0
        [0, 1, 3],  # Face 1
        [1, 2, 3],  # Face 2
        [0, 2, 3]   # Face 3
    ], device=device)  # [4, 3]
    
    faces = elements[:, face_node_indices]  # [M, 4, 3]
    faces_sorted, _ = torch.sort(faces, dim=2)  # [M, 4, 3]
    faces_flat = faces_sorted.view(-1, 3)  # [M * 4, 3]
    
    element_ids = torch.arange(M, device=device).repeat_interleave(4)  # [M*4]
    face_indices = torch.tile(torch.arange(4, device=device), (M,))  # [M*4]
    
    unique_faces, inverse_indices, counts = torch.unique(
        faces_flat, return_inverse=True, return_counts=True, dim=0
    )
    
    shared_mask = counts == 2
    shared_face_ids = torch.nonzero(shared_mask, as_tuple=True)[0]  # [S]
    
    if shared_face_ids.numel() == 0:
        return torch.empty((0, 2, 2), dtype=torch.long, device=device)
    
    sorted_inverse, sorted_order = torch.sort(inverse_indices)
    sorted_element_ids = element_ids[sorted_order]
    sorted_face_indices = face_indices[sorted_order]
    
    positions = torch.searchsorted(sorted_inverse, shared_face_ids)
    
    elem1 = sorted_element_ids[positions]
    face1 = sorted_face_indices[positions]
    elem2 = sorted_element_ids[positions + 1]
    face2 = sorted_face_indices[positions + 1]
    
    shared_face_indices = torch.stack([
        torch.stack([elem1, face1], dim=1),
        torch.stack([elem2, face2], dim=1)
    ], dim=1)  # [S, 2, 2]
    
    return shared_face_indices  # [S, 2, 2]


##### c3d4 노드순서 및 shape function (linear)
'''
Node 0:  (xi, eta, zeta) = (1, 0, 0) 
Node 1:  (xi, eta, zeta) = (0, 1, 0) 
Node 2:  (xi, eta, zeta) = (0, 0, 1) 
Node 3:  (xi, eta, zeta) = (0, 0, 0) 

N_0 = xi 
N_1 = eta 
N_2 = zeta 
N_3 = 1 - xi - eta - zeta 
'''
def c3d4_to_c3d10(coords, elements, rbe2_ids=None, rbe3_ids=None, dtype=torch.float32):
    """
    c3d4 element를 c3d10 element로 변환하는 함수

    Inputs:
        coords (torch.Tensor): Node coordinates [N, 3]
        elements (torch.Tensor): Element connectivity [M, 4]
        rbe2_ids (torch.Tensor): RBE2 constraint node ids [R]
        rbe3_ids (torch.Tensor): RBE3 constraint node ids [R]
        dtype (torch.dtype): Data type (default: torch.float32)
    
    Outputs:
        new_coords (torch.Tensor): New node coordinates [N', 3]
        new_elems (torch.Tensor): New element connectivity [M, 10]
        rbe2_new (torch.Tensor): New RBE2 constraint node ids [R']
        rbe3_new (torch.Tensor): New RBE3 constraint node ids [R']
    """
    if rbe2_ids is None:
        rbe2_ids = set()
    else:
        rbe2_ids = set(rbe2_ids.tolist())
    if rbe3_ids is None:
        rbe3_ids = set()
    else:
        rbe3_ids = set(rbe3_ids.tolist())
    coords_list = coords.tolist()
    edge_to_mid = {}
    rbe2_new = set(rbe2_ids)
    rbe3_new = set(rbe3_ids)
    def get_mid_edge(nA,nB):
        if nA>nB:
            nA,nB=nB,nA
        key=(nA,nB)
        if key in edge_to_mid:
            return edge_to_mid[key]
        cA=coords_list[nA]
        cB=coords_list[nB]
        mid=[(cA[0]+cB[0])/2,(cA[1]+cB[1])/2,(cA[2]+cB[2])/2]
        idx=len(coords_list)
        coords_list.append(mid)
        if nA in rbe2_new and nB in rbe2_new:
            rbe2_new.add(idx)
        if nA in rbe3_new and nB in rbe3_new:
            rbe3_new.add(idx)
        edge_to_mid[key]=idx
        return idx
    new_elems=[]
    for tet in elems:
        n0,n1,n2,n3=tet.tolist()
        c4=get_mid_edge(n0,n1)
        c5=get_mid_edge(n1,n2)
        c6=get_mid_edge(n2,n0)
        c7=get_mid_edge(n0,n3)
        c8=get_mid_edge(n1,n3)
        c9=get_mid_edge(n2,n3)
        new_elems.append([n0,n1,n2,n3,c4,c5,c6,c7,c8,c9])
    return torch.tensor(coords_list,dtype=dtype),torch.tensor(new_elems,dtype=torch.int32),torch.tensor(list(rbe2_new),dtype=torch.int32),torch.tensor(list(rbe3_new),dtype=torch.int32)

def compute_c3d4_B_matrix(coords, elements, device="cuda:0", dtype=torch.float32):
    """
    사면체 상의 강성행렬 B를 구하는 함수.
    strain = B @ displacement
    사면체에선 6*12 형태의 행렬이다.

    Input:
        coords (torch.Tensor): Node coordinates [N, 3]
        elements (torch.Tensor): Element connectivity [M, 4]

    Output:
        torch.Tensor: Strain-displacement matrix [M,6,12]
    """
    coords = coords.to(device=device, dtype=dtype)
    elements = elements.to(device)

    coords_elem = coords[elements]  # [M,4,3]
    M = coords_elem.shape[0]
    ones = torch.ones((M, 4, 1), device=device, dtype=dtype)
    A = torch.cat((ones, coords_elem), dim=2)  # [M,4,4]

    detA = torch.det(A)  # [M]
    if torch.any(detA.abs() < 1e-12):
        raise ValueError("Singular matrix encountered while computing B matrix.")

    invA = torch.inverse(A)  # [M,4,4]

    grads = invA[:, 1:, :]  # [M,3,4]

    dN_dx = grads[:, 0, :]  # [M,4]
    dN_dy = grads[:, 1, :]  # [M,4]
    dN_dz = grads[:, 2, :]  # [M,4]

    B = torch.zeros((M, 6, 12), device=device, dtype=dtype)

    for i in range(4):
        B[:, 0, i*3]     = dN_dx[:, i]
        B[:, 1, i*3+1]   = dN_dy[:, i]
        B[:, 2, i*3+2]   = dN_dz[:, i]
        B[:, 3, i*3]     = dN_dy[:, i]
        B[:, 3, i*3+1]   = dN_dx[:, i]
        B[:, 4, i*3+1]   = dN_dz[:, i]
        B[:, 4, i*3+2]   = dN_dy[:, i]
        B[:, 5, i*3]     = dN_dz[:, i]
        B[:, 5, i*3+2]   = dN_dx[:, i]

    return B # [M,6,12]

def compute_c3d4_K_matrix(coords, elements, E, nu, device="cuda:0", dtype=torch.float32):
    """
    사면체 형상 정보가 입력되면 강성행렬을 사면체 단위로 반환하는 함수

    Input:
        coords (torch.Tensor): Node coordinates [N, 3]
        elements (torch.Tensor): Element connectivity [M, 4]
        E (float): Young's modulus
        nu (float): Poisson's ratio
    
    Output:
        K (torch.Tensor): K matrix [M, 12, 12]
    """
    B = compute_c3d4_B_matrix(coords, elements, device=device, dtype=dtype) # [M, 6, 12]
    D = compute_elasticity_matrix(E, nu, device=device, dtype=dtype) # [6, 6]
    V = compute_tetrahedral_volumes(coords, elements, device=device, dtype=dtype) # [M]

    DB = torch.matmul(D, B) # [M, 6, 12]
    K = torch.matmul(B.transpose(1, 2), DB) # [M, 12, 12]
    K *= V.view(-1, 1, 1) # [M, 12, 12]
    return K # [M, 12, 12]

def compute_c3d4_element_stress(coords, elements, displacement, E, nu, device="cuda:0", dtype=torch.float32):
    """
    최종적으로, 좌표 정보, 사면체 정보, 변위 정보가 주어지면 element단위의 스트레스를 계산하는 함수.
    Young's Modulus와 Poisson Ratio 또한 주어져야한다.

    Input:
        coords (torch.Tensor): Node coordinates [N, 3]
        elements (torch.Tensor): Element connectivity [M, 4]
        displacement (torch.Tensor): Node Displacements [N, 3] 
        E (float): Young's modulus
        nu (float): Poisson's ratio

    Output:
        tuple:
            element_stress_tensor (torch.Tensor): Element-wise stress tensors [M,3,3]
            element_vm_stress (torch.Tensor): Element-wise von Mises stress [M]
    """
    M = elements.shape[0]
    coords = coords.to(device=device, dtype=dtype)
    elements = elements.to(device)
    displacement = displacement.to(device=device, dtype=dtype)

    C = compute_elasticity_matrix(E, nu, device, dtype)  # [6,6]

    disp_elem = displacement[elements].reshape(M, -1)  # [M,12]

    B = compute_c3d4_B_matrix(coords, elements, device, dtype)  # [M,6,12]
    strain = torch.bmm(B, disp_elem.unsqueeze(2)).squeeze(2)  # [M,6]
    stress = torch.matmul(strain, C.t())  # [M,6]
    stress_tensor = compute_stress_tensor(stress)  # [M,3,3]
    element_vm_stress = compute_von_mises_stress(stress_tensor)  # [M]

    return stress_tensor, element_vm_stress # [M,3,3], [M]


##### c3d10 노드순서 및 shape function (quadratic)
'''
Nodes 0, 1, 2, 3: Corner nodes (original tetrahedron).
Nodes 4: between nodes 0 and 1.
Nodes 5: between nodes 1 and 2.
Nodes 6: between nodes 2 and 0.
Nodes 7: between nodes 0 and 3.
Nodes 8: between nodes 1 and 3.
Nodes 9: between nodes 2 and 3.

1.	Corner Nodes (0-3):
N_0 = L_0 (2 L_0 - 1) 
N_1 = L_1 (2 L_1 - 1) 
N_2 = L_2 (2 L_2 - 1) 
N_3 = L_3 (2 L_3 - 1) 
2.	Mid-Side Nodes (4-9):
N_4 = 4 L_0 L_1 
N_5 = 4 L_1 L_2 
N_6 = 4 L_2 L_0 
N_7 = 4 L_0 L_3 
N_8 = 4 L_1 L_3 
N_9 = 4 L_2 L_3 
'''
def c3d10_to_c3d4(c3d10_elements, device="cuda:0"):
    """
    Fully separate C3D10 elements into 8 C3D4 elements to cover the entire volume.

    Args:
        c3d10_elements (torch.Tensor): Tensor of C3D10 elements, shape [M, 10],
                                       where M is the number of elements.

    Returns:
        torch.Tensor: Tensor of C3D4 elements, shape [8*M, 4], where each row
                      corresponds to the 4 nodes of a C3D4 element.
    """
    c3d10_elements = c3d10_elements.to(device)

    sub_tetrahedra = torch.tensor([
        [0, 4, 6, 7],  # Sub-tetrahedron 1
        [4, 1, 5, 8],  # Sub-tetrahedron 2
        [6, 5, 2, 9],  # Sub-tetrahedron 3
        [7, 8, 9, 3],  # Sub-tetrahedron 4
        [4, 6, 7, 5],  # Sub-tetrahedron 5 (mid volume)
        [6, 7, 9, 5],  # Sub-tetrahedron 6 (mid volume)
        [4, 7, 8, 5],  # Sub-tetrahedron 7 (mid volume)
        [5, 8, 7, 9],  # Sub-tetrahedron 8 (mid volume)
    ], dtype=torch.long, device=device)  # [8, 4]

    M = c3d10_elements.shape[0]

    c3d4_elements = c3d10_elements[:, sub_tetrahedra]  # Shape [M, 8, 4]
    c3d4_elements = c3d4_elements.view(-1, 4)  # Flatten to shape [8*M, 4]

    return c3d4_elements # [8*M, 4]

def c3d10_integration_points(device="cuda:0", dtype=torch.float32):
    c3d10_11_integration_points = torch.tensor([
        [0.25, 0.25, 0.25],      # Centroid
        [0.1, 0.1, 0.1],
        [0.1, 0.1, 0.7],
        [0.1, 0.7, 0.1],
        [0.7, 0.1, 0.1],
        [0.1, 0.4, 0.4],
        [0.4, 0.1, 0.4],
        [0.4, 0.4, 0.1],
        [0.3, 0.3, 0.3],
        [0.2, 0.2, 0.6],
        [0.2, 0.6, 0.2]
    ], dtype=dtype)

    c3d10_11_weights = torch.tensor([
        0.1,   # Weight for centroid
        0.05,  # Weights for points near the corners
        0.05,
        0.05,
        0.05,
        0.03,  # Weights for mid-edge points
        0.03,
        0.03,
        0.02,  # Weight for the central point
        0.02,
        0.02
    ], dtype=dtype)

    return c3d10_11_integration_points.to(device), c3d10_11_weights.to(device) # [11, 3], [11]

def compute_c3d10_Jacobian(coords, elements, integral_point, device="cuda:0", dtype=torch.float32):
    """
    주어진 적분점에서 각 c3d10 사면체에 대해 야코비안을 구하는 함수

    Input:
        coords (torch.Tensor): Node coordinates [N, 3]
        elements (torch.Tensor): Element connectivity [M, 10]
        integral_point (torch.Tensor): Natural coordinates [xi, eta, zeta]

    Output:
        Jacobian (torch.Tensor): Jacobian matrices [M, 3, 3]
    """
    coords = coords.to(device=device, dtype=dtype)
    elements = elements.to(device)
    xi, eta, zeta = integral_point.to(device=device, dtype=dtype)

    dN_dnatural = torch.tensor([
        [4 * xi - 1, 0, 0],                             # Node 0
        [0, 4 * eta - 1, 0],                           # Node 1
        [0, 0, 4 * zeta - 1],                          # Node 2
        [-4 * (1 - xi - eta - zeta) + 1,               # Node 3
         -4 * (1 - xi - eta - zeta) + 1,
         -4 * (1 - xi - eta - zeta) + 1],
        [4 * eta, 4 * xi, 0],                          # Node 4
        [0, 4 * zeta, 4 * eta],                        # Node 5
        [4 * zeta, 0, 4 * xi],                         # Node 6
        [4 * (1 - 2 * xi - eta - zeta), -4 * xi, -4 * xi], # Node 7
        [-4 * eta, 4 * (1 - xi - 2 * eta - zeta), -4 * eta], # Node 8
        [-4 * zeta, -4 * zeta, 4 * (1 - xi - eta - 2 * zeta)] # Node 9
    ]).to(device=device, dtype=dtype)  # (10, 3)

    element_coords = coords[elements]  # (M, 10, 3)
    jacobian = torch.einsum("ji,mjk->mik", dN_dnatural, element_coords)  # (M, 3, 3)

    return jacobian # [M, 3, 3]

def compute_c3d10_shape_gradients(coords, elements, integral_point, device="cuda:0", dtype=torch.float32):
    """
    주어진 적분점에서 각 c3d10 사면체에 대해 shape function gradient를 구하는 함수

    Input:
        coords (torch.Tensor): Node coordinates [N, 3]
        elements (torch.Tensor): Element connectivity [M, 10]
        integral_point (torch.Tensor): Natural coordinates [xi, eta, zeta]

    Output:
        dN_global (torch.Tensor): Shape function gradients in global coordinates [M, 10, 3]
    """
    jacobian = compute_c3d10_Jacobian(coords, elements, integral_point, device=device, dtype=dtype)  # (M, 3, 3)
    jacobian_inv = torch.inverse(jacobian)  # (M, 3, 3)

    xi, eta, zeta = integral_point.to(device=device, dtype=dtype)
    dN_dnatural = torch.tensor([
        [4 * xi - 1, 0, 0],                             # Node 0
        [0, 4 * eta - 1, 0],                           # Node 1
        [0, 0, 4 * zeta - 1],                          # Node 2
        [-4 * (1 - xi - eta - zeta) + 1,               # Node 3
         -4 * (1 - xi - eta - zeta) + 1,
         -4 * (1 - xi - eta - zeta) + 1],
        [4 * eta, 4 * xi, 0],                          # Node 4
        [0, 4 * zeta, 4 * eta],                        # Node 5
        [4 * zeta, 0, 4 * xi],                         # Node 6
        [4 * (1 - 2 * xi - eta - zeta), -4 * xi, -4 * xi], # Node 7
        [-4 * eta, 4 * (1 - xi - 2 * eta - zeta), -4 * eta], # Node 8
        [-4 * zeta, -4 * zeta, 4 * (1 - xi - eta - 2 * zeta)] # Node 9
    ]).to(device=device, dtype=dtype)  # (10, 3)

    dN_global = torch.einsum("mij,nj->mni", jacobian_inv, dN_dnatural)  # (M, 10, 3)

    return dN_global # [M, 10, 3]

def compute_c3d10_B_matrix(coords, elements, integral_point, device="cuda:0", dtype=torch.float32):
    """
    주어진 적분점에서 각 c3d10 사면체에 대해 B 행렬을 구하는 함수

    Input:
        coords (torch.Tensor): Node coordinates [N, 3]
        elements (torch.Tensor): Element connectivity [M, 10]
        integral_point (torch.Tensor): Natural coordinates [xi, eta, zeta]

    Output:
        B_matrices (torch.Tensor): B matrices for all elements [M, 6, 30]
    """
    dN_global = compute_c3d10_shape_gradients(coords, elements, integral_point, device=device, dtype=dtype) # [M, 10, 3]

    M = elements.shape[0] 
    B_matrices = torch.zeros((M, 6, 30), device=device, dtype=dtype)  # [M, 6, 30]

    for i in range(10):
        B_matrices[:, 0, 3 * i + 0] = dN_global[:, i, 0]  # ∂N/∂x
        B_matrices[:, 1, 3 * i + 1] = dN_global[:, i, 1]  # ∂N/∂y
        B_matrices[:, 2, 3 * i + 2] = dN_global[:, i, 2]  # ∂N/∂z
        B_matrices[:, 3, 3 * i + 0] = dN_global[:, i, 1]  # ∂N/∂y
        B_matrices[:, 3, 3 * i + 1] = dN_global[:, i, 0]  # ∂N/∂x
        B_matrices[:, 4, 3 * i + 1] = dN_global[:, i, 2]  # ∂N/∂z
        B_matrices[:, 4, 3 * i + 2] = dN_global[:, i, 1]  # ∂N/∂y
        B_matrices[:, 5, 3 * i + 0] = dN_global[:, i, 2]  # ∂N/∂z
        B_matrices[:, 5, 3 * i + 2] = dN_global[:, i, 0]  # ∂N/∂x

    return B_matrices

def compute_c3d10_element_stress(coords, elements, displacement, E, nu, integral_point=None, single=True, device="cuda:0", dtype=torch.float32):
    """
    최종적으로, 좌표 정보, 사면체 정보, 변위 정보가 주어지면 element단위의 스트레스를 적분점별로 계산하는 함수.
    Young's Modulus와 Poisson Ratio 또한 주어져야한다.
    적분점도 주어지며, [n_int, 4] 형태이다. None이면 기본 11점 integration point를 사용한다.
    각 element 별로 1개, 혹은 적분점 별로 1개를 뽑을 수 있다. single=True면 적분점별로 1개

    Input:
        coords (torch.Tensor): Node coordinates [N, 3]
        elements (torch.Tensor): Element connectivity [M, 10]
        displacement (torch.Tensor): Node Displacements [N, 3]
        E (float): Young's modulus
        nu (float): Poisson's ratio
        integral point (torch.Tensor): natural coordinates [xi, eta, zeta, weight], [n_int, 4]


    Output:
        tuple:
            stress_per_ip (torch.Tensor): Ip-Element-wise stress tensors [M,n_int,3,3]
            vm_stress_per_ip (torch.Tensor): Ip-Element-wise von Mises stress [M,n_int]
    """
    M = elements.shape[0]
    coords = coords.to(device=device, dtype=dtype)
    elements = elements.to(device)
    displacement = displacement.to(device=device, dtype=dtype)

    if integral_point == None:
        p, w = c3d10_integration_points(device=device, dtype=dtype)
    else:
        p, w = integral_point[:,:3].to(device=device, dtype=dtype), integral_point[:,-1].to(device=device, dtype=dtype)
    
    D = compute_elasticity_matrix(E, nu, device=device, dtype=dtype)
    disp_elem = displacement[elements].reshape(M, -1)  # [M, 30]

    if single:
        stress_result = torch.zeros((M, 3, 3), device=device, dtype=dtype)
        vm_stress_result = torch.zeros((M), device=device, dtype=dtype)
    else:
        stress_result = []
        vm_stress_result = []

    for idx in range(p.shape[0]):
        ip = p[idx]
        
        B = compute_c3d10_B_matrix(coords, elements, ip, device=device, dtype=dtype) # [M, 6, 30]
        
        strain = torch.bmm(B, disp_elem.unsqueeze(2)).squeeze(2)  # [M,6]
        stress = torch.matmul(strain, D.t())  # [M,6]

        stress_tensor = compute_stress_tensor(stress)  # [M,3,3]
        element_vm_stress = compute_von_mises_stress(stress_tensor)  # [M]

        if single:
            stress_result += stress_tensor * w[idx]
            vm_stress_result += element_vm_stress * w[idx]
        else:
            stress_result.append(stress_tensor)
            vm_stress_result.append(element_vm_stress)

    if single:
        return stress_result, vm_stress_result # [M,3,3], [M]
    else:
        return torch.tensor(np.array(stress_result), device=device, dtype=dtype), torch.tensor(np.array(vm_stress_result), device=device, dtype=dtype) # [n_int, M, 3, 3], [n_int, M]

def compute_c3d10_K_matrix(coords, elements, E, nu, integral_point=None, single=True, device="cuda:0", dtype=torch.float32):
    """
    Computes the element stiffness matrix for each C3D10 element.
    각 element 별로 1개, 혹은 적분점 별로 1개를 뽑을 수 있다. single=True면 적분점별로 1개

    Input:
        coords (torch.Tensor): Node coordinates [N, 3].
        elements (torch.Tensor): Element connectivity [M, 10].
        E (float): Young's modulus.
        nu (float): Poisson's ratio.
        integral point (torch.Tensor): natural coordinates [xi, eta, zeta, weight], [n_int, 4]
        single (bool)

    Output:
        K_elements (torch.Tensor): Stiffness matrices for all elements [M, 30, 30] / [M, n_int, 30, 30]
    """
    M = elements.shape[0]
    coords = coords.to(device=device, dtype=dtype)
    elements = elements.to(device)
    if single:
        K_elements = torch.zeros((M, 30, 30), device=device, dtype=dtype)
    else:
        K_elements = []

    if integral_point == None:
        integration_points, weights = c3d10_integration_points(device=device, dtype=dtype)  # [11, 3], [11]
    else:
        integration_points, weights = integral_point[:,:3].to(device=device, dtype=dtype), integral_point[:,-1].to(device=device, dtype=dtype)

    D = compute_elasticity_matrix(E, nu, device=device, dtype=dtype)  # [6, 6]

    for ip_idx in range(integration_points.shape[0]):
        ip = integration_points[ip_idx]
        weight = weights[ip_idx]

        B = compute_c3d10_B_matrix(coords, elements, ip, device=device, dtype=dtype)  # [M, 6, 30]
        J = compute_c3d10_Jacobian(coords, elements, ip, device=device, dtype=dtype)  # [M, 3, 3]
        detJ = torch.det(J)  # [M]
        BD = torch.einsum("mji,jk->mik", B, D) # [M, 6, 30]
        K_ip = torch.einsum("mik,mkj->mij", BD, B)  # [M, 30, 30]
        if single:
            K_elements += K_ip * detJ.view(M, 1, 1) * weight
        else:
            K_elements.append(K_ip*detJ.view(M, 1, 1))

    if single:
        return K_elements  # [M, 30, 30]
    else:
        return torch.tensor(np.array(K_elements), device=device, dtype=dtype) # [n_int, M, 30, 30]


#################################################################################################
########################################   Hexahedral   #########################################
#################################################################################################


##### 기초연산 (c3d8, c3d20 모두 사용)
def compute_hexahedral_volumes(coords, elements, device="cuda:0", dtype=torch.float32):
    """
    육면체 부피 구하기 

    Input:
        coords (torch.Tensor): Node coordinates [N, 3]
        elements (torch.Tensor): Element connectivity [M, 8]
    
    Output:
        volumes (torch.Tensor): Hexahedral volumes [M]
    """
    coords = coords.to(device=device, dtype=dtype)
    elements = elements.to(device)

    p1 = coords[elements[:, 0]]  # Node 0
    p2 = coords[elements[:, 1]]  # Node 1
    p3 = coords[elements[:, 2]]  # Node 2
    p4 = coords[elements[:, 3]]  # Node 3
    p5 = coords[elements[:, 4]]  # Node 4
    p6 = coords[elements[:, 5]]  # Node 5
    p7 = coords[elements[:, 6]]  # Node 6
    p8 = coords[elements[:, 7]]  # Node 7

    def v(p1, p2, p3, p4):
        v1 = p2 - p1
        v2 = p3 - p1
        v3 = p4 - p1

        det = torch.abs(torch.det(torch.stack([v1, v2, v3], dim=1)))

        volume = det / 6.0 

        return volume

    v1 = v(p1, p2, p4, p5)  # Tetrahedron 1
    v2 = v(p2, p3, p4, p7)  # Tetrahedron 2
    v3 = v(p2, p4, p5, p6)  # Tetrahedron 3
    v4 = v(p4, p5, p6, p8)  # Tetrahedron 4
    v5 = v(p4, p6, p7, p8)  # Tetrahedron 5
    v6 = v(p4, p6, p7, p2)  # Tetrahedron 6

    volumes = v1 + v2 + v3 + v4 + v5 + v6

    return volumes # [M]

def compute_hexahedral_surface_faces_with_extra_node(elements, device="cuda:0"):
    """
    육면체 정보가 주어지면 겉면을 이루는 면과, 면을 이루지 않는 노드 번호를 구하는 함수.
    면을 이루지 않는 노드 정보는 법선을 정렬할 때 사용된다.

    Input:
        elements (torch.Tensor): Element connectivity [M, 8]
        device (str): Device to perform computations on ("cuda:0" or "cpu")
    
    Returns:
        surface_faces (torch.Tensor): Surface face connectivity [K, 4]
        extra_nodes (torch.Tensor): One of the nodes not forming the face [K]
    """
    elements = elements.to(device)

    faces = torch.cat([
        elements[:, [0, 1, 5, 4]],  # Face opposite to nodes [2, 3, 6, 7]
        elements[:, [1, 2, 6, 5]],  # Face opposite to nodes [0, 3, 4, 7]
        elements[:, [2, 3, 7, 6]],  # Face opposite to nodes [0, 1, 4, 5]
        elements[:, [0, 4, 7, 3]],  # Face opposite to nodes [1, 2, 5, 6]
        elements[:, [0, 3, 2, 1]],  # Face opposite to nodes [4, 5, 6, 7]
        elements[:, [4, 5, 6, 7]],  # Face opposite to nodes [0, 1, 2, 3]
    ], dim=0)

    extra_nodes = torch.cat([
        elements[:, 2],  # Extra node for face [0, 1, 5, 4]
        elements[:, 0],  # Extra node for face [1, 2, 6, 5]
        elements[:, 0],  # Extra node for face [2, 3, 7, 6]
        elements[:, 1],  # Extra node for face [0, 4, 7, 3]
        elements[:, 4],  # Extra node for face [0, 3, 2, 1]
        elements[:, 0],  # Extra node for face [4, 5, 6, 7]
    ], dim=0)

    sorted_faces, indices = torch.sort(faces, dim=1)

    unique_faces, inverse_indices, counts = torch.unique(sorted_faces, dim=0, return_inverse=True, return_counts=True)

    surface_face_mask = counts[inverse_indices] == 1
    surface_faces = faces[surface_face_mask]
    surface_extra_nodes = extra_nodes[surface_face_mask]

    return surface_faces, surface_extra_nodes  # [K, 4], [K]

def compute_hexahedral_surface_normals(coords, elements, device="cuda:0", dtype=torch.float32):
    """
    형상 바깥을 향하는 법선을 일괄적으로 구하는 함수 (Hexahedral Surface Normals)
    겉면의 "extra node" 정보가 필요하다.

    Input:
        coords (torch.Tensor): Node coordinates [N, 3]
        elements (torch.Tensor): Element connectivity [M, 8]

    Output:
        surface_normals (torch.Tensor): Normal vectors of surface faces [K, 3]
    """
    coords = coords.to(device=device, dtype=dtype)
    elements = elements.to(device)

    surface_faces, extra_nodes = compute_hexahedral_surface_faces_with_extra_node(elements, device=device)

    p1 = coords[surface_faces[:, 0]]  # [K, 3]
    p2 = coords[surface_faces[:, 1]]  # [K, 3]
    p3 = coords[surface_faces[:, 2]]  # [K, 3]
    p4 = coords[surface_faces[:, 3]]  # [K, 3]

    v1 = p2 - p1  # [K, 3]
    v2 = p3 - p1  # [K, 3]

    normals = torch.cross(v1, v2, dim=1)  # [K, 3]
    normals = normals / torch.norm(normals, dim=1, keepdim=True) 

    face_centers = (p1 + p2 + p3 + p4) / 4  # [K, 3]

    extra_points = coords[extra_nodes]  # [K, 3]

    to_extra = extra_points - face_centers  # [K, 3]
    to_extra_normalized = to_extra / torch.norm(to_extra, dim=1, keepdim=True) 

    dot_product = (normals * to_extra_normalized).sum(dim=1)  
    normals[dot_product > 0] = -normals[dot_product > 0]  

    return normals

def compute_hexahedral_node_curvatures(coords, elements, device="cuda:0", dtype=torch.float32):
    """
    법선 정보와 좌표 개수를 입력받아 노드별 곡률을 구하는 함수 (Hexahedral)

    Input:
        coords (torch.Tensor): Node coordinates [N, 3]
        elements (torch.Tensor): Element connectivity [M, 8]

    Output:
        curv (torch.Tensor): Tensor of mean curvatures along nodes [N, 3]
    """
    coords = coords.to(device=device, dtype=dtype)
    elements = elements.to(device)

    N = coords.shape[0]
    surface_faces, _ = compute_hexahedral_surface_faces_with_extra_node(elements, device=device)
    surface_normals = compute_hexahedral_surface_normals(coords, elements, device=device, dtype=dtype)  # [K, 3]

    accumulated_curvatures = torch.zeros((N, 3), device=device)  # [N, 3]
    node_counts = torch.zeros(N, device=device)  # [N]

    face_normals_expanded = face_normals[:, None, :].expand(-1, 4, -1)  # [K, 4, 3]

    accumulated_curvatures.scatter_add_(
        0,
        surface_faces.view(-1, 1).expand(-1, 3), 
        face_normals_expanded.reshape(-1, 3)    
    )

    ones = torch.ones(surface_faces.numel(), device=device)  # [4K]
    node_counts.scatter_add_(
        0,
        surface_faces.view(-1),  
        ones                    
    )

    node_counts[node_counts == 0] = 1

    mean_curvature_vectors = accumulated_curvatures / node_counts.view(-1, 1)  # [N, 3]

    return mean_curvature_vectors

def compute_hexahedral_normals_and_area(coords, elements, device="cuda:0", dtype=torch.float32):
    """
    각 육면체 element의 면별 법선을 구하고, 면적을 계산하는 함수
    법선 벡터는 면적에 비례하며, 바깥을 향하도록 정렬된다.
    면의 순서는 아래와 같다:
    [0, 1, 5, 4],  # Face 0
    [1, 2, 6, 5],  # Face 1
    [2, 3, 7, 6],  # Face 2
    [0, 4, 7, 3],  # Face 3
    [0, 3, 2, 1],  # Face 4 (Bottom face)
    [4, 5, 6, 7]   # Face 5 (Top face)

    Inputs:
        coords (torch.Tensor): Node coordinates [N, 3]
        elements (torch.Tensor): Element connectivity [M, 8] / [M, 20]
        
    Outputs:
        normal_vectors (torch.Tensor): Normal vectors for each face of each element [M, 6, 3]
    """
    coords = coords.to(device=device, dtype=dtype)
    elements = elements.to(device)
    

    M = elements.shape[0]
    normal_vectors = torch.zeros((M, 6, 3), device=device, dtype=dtype)

    coords_elem = coords[elements]  # [M, 8, 3]

    faces = torch.tensor([
        [0, 1, 5, 4],  # Face 0
        [1, 2, 6, 5],  # Face 1
        [2, 3, 7, 6],  # Face 2
        [0, 4, 7, 3],  # Face 3
        [0, 3, 2, 1],  # Face 4 (Bottom face)
        [4, 5, 6, 7]   # Face 5 (Top face)
    ], device=device)  # [6, 4]

    face_coords = coords_elem[:, faces]  # [M, 6, 4, 3]
    centroids = face_coords.mean(dim=2)  # [M, 6, 3]

    edge1 = face_coords[:, :, 1, :] - face_coords[:, :, 0, :]  # [M, 6, 3]
    edge2 = face_coords[:, :, 3, :] - face_coords[:, :, 0, :]  # [M, 6, 3]

    normal_vectors = torch.cross(edge1, edge2, dim=2)  # [M, 6, 3]

    extra_nodes = torch.tensor([2, 0, 0, 1, 6, 0], device=device)  
    extra_node_coords = coords[elements[:, extra_nodes]]  # [M, 6, 3]

    centroid_to_extra = extra_node_coords - centroids  # [M, 6, 3]
    dot_products = torch.sum(normal_vectors * centroid_to_extra, dim=2)  # [M, 6]

    flip_mask = dot_products > 0
    normal_vectors[flip_mask] = -normal_vectors[flip_mask]

    return normal_vectors # [M, 6, 3]

def identify_hexahedral_shared_faces(elements, device="cuda:0"):
    """
    elements 정보가 주어지면 각 면이 공유되는 element 정보를 구하는 함수 (Hexahedral)
    아웃풋은 [S, 2, 2]이며, 총 S개의 면에 대해 [2, 2] 연결 정보를 제공한다.
    [2, 2]: [[element id, face index], [element id, face index]]

    Inputs:
        elements (torch.Tensor): Element connectivity [M, 8] / [M, 20]
    
    Outputs:
        shared_face_indices (torch.Tensor): Indices of elements sharing each face [S, 2, 2]
    """
    M = elements.shape[0]
    elements = elements.to(device)

    face_node_indices = torch.tensor([
        [0, 1, 5, 4],  # Face 0
        [1, 2, 6, 5],  # Face 1
        [2, 3, 7, 6],  # Face 2
        [0, 4, 7, 3],  # Face 3
        [0, 3, 2, 1],  # Face 4 (Bottom face)
        [4, 5, 6, 7]   # Face 5 (Top face)
    ], device=device)  # [6, 4]

    faces = elements[:, face_node_indices]  # [M, 6, 4]

    faces_sorted, _ = torch.sort(faces, dim=2)  # [M, 6, 4]
    faces_flat = faces_sorted.view(-1, 4)  # [M * 6, 4]

    element_ids = torch.arange(M, device=device).repeat_interleave(6)  # [M*6]
    face_indices = torch.tile(torch.arange(6, device=device), (M,))  # [M*6]

    unique_faces, inverse_indices, counts = torch.unique(
        faces_flat, return_inverse=True, return_counts=True, dim=0
    )

    shared_mask = counts == 2
    shared_face_ids = torch.nonzero(shared_mask, as_tuple=True)[0]  # [S]

    if shared_face_ids.numel() == 0:
        return torch.empty((0, 2, 2), dtype=torch.long, device=device)

    sorted_inverse, sorted_order = torch.sort(inverse_indices)
    sorted_element_ids = element_ids[sorted_order]
    sorted_face_indices = face_indices[sorted_order]

    positions = torch.searchsorted(sorted_inverse, shared_face_ids)

    elem1 = sorted_element_ids[positions]
    face1 = sorted_face_indices[positions]
    elem2 = sorted_element_ids[positions + 1]
    face2 = sorted_face_indices[positions + 1]

    shared_face_indices = torch.stack([
        torch.stack([elem1, face1], dim=1),
        torch.stack([elem2, face2], dim=1)
    ], dim=1)  # [S, 2, 2]

    return shared_face_indices  # [S, 2, 2]


##### c3d8 노드순서 및 shape function (linear)
'''
Node 0: (-1, -1, -1)
Node 1: ( 1, -1, -1)
Node 2: ( 1,  1, -1)
Node 3: (-1,  1, -1)
Node 4: (-1, -1,  1)
Node 5: ( 1, -1,  1)
Node 6: ( 1,  1,  1)
Node 7: (-1,  1,  1)

N_0 = (1 - ξ)(1 - η)(1 - ζ)/8
N_1 = (1 + ξ)(1 - η)(1 - ζ)/8
N_2 = (1 + ξ)(1 + η)(1 - ζ)/8
N_3 = (1 - ξ)(1 + η)(1 - ζ)/8
N_4 = (1 - ξ)(1 - η)(1 + ζ)/8
N_5 = (1 + ξ)(1 - η)(1 + ζ)/8
N_6 = (1 + ξ)(1 + η)(1 + ζ)/8
N_7 = (1 - ξ)(1 + η)(1 + ζ)/8
'''
def c3d8_to_c3d4(c3d8_elements, device="cuda:0"):
    """
    Fully separate C3D8 elements into 6 C3D4 elements to cover the entire volume.

    Input:
        c3d8_elements (torch.Tensor): Tensor of C3D8 elements [M, 8]

    Output:
        c3d4_elements (torch.Tensor): C3D4 elements [6*M, 4]
    """
    c3d8_elements = c3d8_elements.to(device)

    sub_tetrahedra = torch.tensor([
        [0, 1, 3, 4],  # Tetrahedron 1
        [1, 2, 3, 6],  # Tetrahedron 2
        [1, 3, 4, 5],  # Tetrahedron 3
        [3, 4, 5, 7],  # Tetrahedron 4
        [3, 5, 6, 7],  # Tetrahedron 5
        [3, 5, 6, 2],  # Tetrahedron 6
    ], dtype=torch.long, device=device)  # [6, 4]

    M = c3d8_elements.shape[0]

    c3d4_elements = c3d8_elements[:, sub_tetrahedra]  # [M, 6, 4]
    c3d4_elements = c3d4_elements.view(-1, 4)  # [6*M, 4]

    return c3d4_elements # [6*M, 4]

def c3d8_integration_points(device="cuda:0", dtype=torch.float32):
    one_over_sqrt3 = 1.0 / torch.sqrt(torch.tensor(3.0, dtype=dtype, device=device))
    
    points = torch.tensor([
        [-one_over_sqrt3, -one_over_sqrt3, -one_over_sqrt3],
        [-one_over_sqrt3, -one_over_sqrt3,  one_over_sqrt3],
        [-one_over_sqrt3,  one_over_sqrt3, -one_over_sqrt3],
        [-one_over_sqrt3,  one_over_sqrt3,  one_over_sqrt3],
        [ one_over_sqrt3, -one_over_sqrt3, -one_over_sqrt3],
        [ one_over_sqrt3, -one_over_sqrt3,  one_over_sqrt3],
        [ one_over_sqrt3,  one_over_sqrt3, -one_over_sqrt3],
        [ one_over_sqrt3,  one_over_sqrt3,  one_over_sqrt3]
    ], dtype=dtype, device=device)
    
    weights = torch.ones(8, dtype=dtype, device=device)
    
    return points, weights # [8, 3], [8]

def compute_c3d8_Jacobian(coords, elements, integral_point, device="cuda:0", dtype=torch.float32):
    """
    주어진 적분점에서 각 c3d8 육면체에 대해 야코비안을 구하는 함수

    Input:
        coords (torch.Tensor): Node coordinates [N, 3]
        elements (torch.Tensor): Element connectivity [M, 8]
        integral_point (torch.Tensor): Natural coordinates [xi, eta, zeta]

    Output:
        Jacobian (torch.Tensor): Jacobian matrices [M, 3, 3]
    """
    coords = coords.to(device=device, dtype=dtype)
    elements = elements.to(device)
    xi, eta, zeta = integral_point.to(device=device, dtype=dtype)

    dN_dnatural = torch.tensor([
        [-0.125 * (1 - eta) * (1 - zeta), -0.125 * (1 - xi) * (1 - zeta), -0.125 * (1 - xi) * (1 - eta)],  # Node 0
        [ 0.125 * (1 - eta) * (1 - zeta), -0.125 * (1 + xi) * (1 - zeta), -0.125 * (1 + xi) * (1 - eta)],  # Node 1
        [ 0.125 * (1 + eta) * (1 - zeta),  0.125 * (1 + xi) * (1 - zeta), -0.125 * (1 + xi) * (1 + eta)],  # Node 2
        [-0.125 * (1 + eta) * (1 - zeta),  0.125 * (1 - xi) * (1 - zeta), -0.125 * (1 - xi) * (1 + eta)],  # Node 3
        [-0.125 * (1 - eta) * (1 + zeta), -0.125 * (1 - xi) * (1 + zeta),  0.125 * (1 - xi) * (1 - eta)],  # Node 4
        [ 0.125 * (1 - eta) * (1 + zeta), -0.125 * (1 + xi) * (1 + zeta),  0.125 * (1 + xi) * (1 - eta)],  # Node 5
        [ 0.125 * (1 + eta) * (1 + zeta),  0.125 * (1 + xi) * (1 + zeta),  0.125 * (1 + xi) * (1 + eta)],  # Node 6
        [-0.125 * (1 + eta) * (1 + zeta),  0.125 * (1 - xi) * (1 + zeta),  0.125 * (1 - xi) * (1 + eta)]   # Node 7
    ]).to(device=device, dtype=dtype)  # (8, 3)

    element_coords = coords[elements]  # (M, 8, 3)

    jacobian = torch.einsum("ji,mjk->mik", dN_dnatural, element_coords)  # (M, 3, 3)

    return jacobian # (M, 3, 3)

def compute_c3d8_shape_gradients(coords, elements, integral_point, device="cuda:0", dtype=torch.float32):
    """
    주어진 적분점에서 각 c3d8 육면체에 대해 shape function gradient를 구하는 함수

    Input:
        coords (torch.Tensor): Node coordinates [N, 3]
        elements (torch.Tensor): Element connectivity [M, 8]
        integral_point (torch.Tensor): Natural coordinates [xi, eta, zeta]

    Output:
        dN_global (torch.Tensor): Shape function gradients in global coordinates [M, 8, 3]
    """
    jacobian = compute_c3d8_Jacobian(coords, elements, integral_point, device=device, dtype=dtype)  # (M, 3, 3)

    jacobian_inv = torch.inverse(jacobian)  # (M, 3, 3)

    xi, eta, zeta = integral_point.to(device=device, dtype=dtype)
    dN_dnatural = torch.tensor([
        [-0.125 * (1 - eta) * (1 - zeta), -0.125 * (1 - xi) * (1 - zeta), -0.125 * (1 - xi) * (1 - eta)],  # Node 0
        [ 0.125 * (1 - eta) * (1 - zeta), -0.125 * (1 + xi) * (1 - zeta), -0.125 * (1 + xi) * (1 - eta)],  # Node 1
        [ 0.125 * (1 + eta) * (1 - zeta),  0.125 * (1 + xi) * (1 - zeta), -0.125 * (1 + xi) * (1 + eta)],  # Node 2
        [-0.125 * (1 + eta) * (1 - zeta),  0.125 * (1 - xi) * (1 - zeta), -0.125 * (1 - xi) * (1 + eta)],  # Node 3
        [-0.125 * (1 - eta) * (1 + zeta), -0.125 * (1 - xi) * (1 + zeta),  0.125 * (1 - xi) * (1 - eta)],  # Node 4
        [ 0.125 * (1 - eta) * (1 + zeta), -0.125 * (1 + xi) * (1 + zeta),  0.125 * (1 + xi) * (1 - eta)],  # Node 5
        [ 0.125 * (1 + eta) * (1 + zeta),  0.125 * (1 + xi) * (1 + zeta),  0.125 * (1 + xi) * (1 + eta)],  # Node 6
        [-0.125 * (1 + eta) * (1 + zeta),  0.125 * (1 - xi) * (1 + zeta),  0.125 * (1 - xi) * (1 + eta)]   # Node 7
    ]).to(device=device, dtype=dtype)  # (8, 3)

    dN_global = torch.einsum("mij,nj->mni", jacobian_inv, dN_dnatural)  # (M, 8, 3)

    return dN_global # (M, 8, 3)

def compute_c3d8_B_matrix(coords, elements, integral_point, device="cuda:0", dtype=torch.float32):
    """
    주어진 적분점에서 각 c3d8 육면체에 대해 B 행렬을 구하는 함수

    Input:
        coords (torch.Tensor): Node coordinates [N, 3]
        elements (torch.Tensor): Element connectivity [M, 8]
        integral_point (torch.Tensor): Natural coordinates [xi, eta, zeta]

    Output:
        B_matrices (torch.Tensor): B matrices for all elements [M, 6, 24]
    """
    dN_global = compute_c3d8_shape_gradients(coords, elements, integral_point, device=device, dtype=dtype)  # [M, 8, 3]

    M = elements.shape[0]  
    B_matrices = torch.zeros((M, 6, 24), device=device, dtype=dtype)  # [M, 6, 24]

    for i in range(8): 
        B_matrices[:, 0, 3 * i + 0] = dN_global[:, i, 0]  # ∂N/∂x
        B_matrices[:, 1, 3 * i + 1] = dN_global[:, i, 1]  # ∂N/∂y
        B_matrices[:, 2, 3 * i + 2] = dN_global[:, i, 2]  # ∂N/∂z
        B_matrices[:, 3, 3 * i + 0] = dN_global[:, i, 1]  # ∂N/∂y
        B_matrices[:, 3, 3 * i + 1] = dN_global[:, i, 0]  # ∂N/∂x
        B_matrices[:, 4, 3 * i + 1] = dN_global[:, i, 2]  # ∂N/∂z
        B_matrices[:, 4, 3 * i + 2] = dN_global[:, i, 1]  # ∂N/∂y
        B_matrices[:, 5, 3 * i + 0] = dN_global[:, i, 2]  # ∂N/∂z
        B_matrices[:, 5, 3 * i + 2] = dN_global[:, i, 0]  # ∂N/∂x

    return B_matrices

def compute_c3d8_element_stress(coords, elements, displacement, E, nu, integral_point=None, single=True, device="cuda:0", dtype=torch.float32):
    """
    Compute element-wise stresses and von Mises stresses for C3D8 elements without using lists.

    Args:
        coords (torch.Tensor): Node coordinates [N, 3]
        elements (torch.Tensor): Element connectivity [M, 8]
        displacement (torch.Tensor): Node displacements [N, 3]
        E (float): Young's modulus
        nu (float): Poisson's ratio
        integral_point (torch.Tensor): Natural coordinates [xi, eta, zeta, weight], [n_int, 4]

    Returns:
        tuple:
            stress_per_ip (torch.Tensor): Ip-Element-wise stress tensors [M, n_int, 3, 3]
            vm_stress_per_ip (torch.Tensor): Ip-Element-wise von Mises stress [M, n_int]
    """
    M = elements.shape[0]
    coords = coords.to(device=device, dtype=dtype)
    elements = elements.to(device)
    displacement = displacement.to(device=device, dtype=dtype)

    if integral_point is None:
        p, w = c3d8_integration_points(device=device, dtype=dtype)
    else:
        p, w = integral_point[:, :3].to(device=device, dtype=dtype), integral_point[:, -1].to(device=device, dtype=dtype)

    n_int = p.shape[0] 

    D = compute_elasticity_matrix(E, nu, device=device, dtype=dtype)

    disp_elem = displacement[elements].reshape(M, -1)  # [M, 24]

    stress_result = torch.zeros((M, n_int, 3, 3), device=device, dtype=dtype)  # [M, n_int, 3, 3]
    vm_stress_result = torch.zeros((M, n_int), device=device, dtype=dtype)    # [M, n_int]

    for idx in range(n_int):
        ip = p[idx]

        B = compute_c3d8_B_matrix(coords, elements, ip, device=device, dtype=dtype)  # [M, 6, 24]

        strain = torch.bmm(B, disp_elem.unsqueeze(2)).squeeze(2)  # [M, 6]

        stress = torch.matmul(strain, D.t())  # [M, 6]

        stress_tensor = compute_stress_tensor(stress)  # [M, 3, 3]

        element_vm_stress = compute_von_mises_stress(stress_tensor)  # [M]

        stress_result[:, idx, :, :] = stress_tensor  # [M, n_int, 3, 3]
        vm_stress_result[:, idx] = element_vm_stress  # [M, n_int]

    if single:
        stress_result = torch.einsum("i,mijk->mjk", w, stress_result)  # [M, 3, 3]
        vm_stress_result = torch.einsum("i,mi->m", w, vm_stress_result)  # [M]

    return stress_result, vm_stress_result # [M, n_int, 3, 3], [M, n_int] or [M, 3, 3], [M]

def compute_c3d8_K_matrix(coords, elements, E, nu, integral_point=None, single=True, device="cuda:0", dtype=torch.float32):
    """
    Computes the element stiffness matrix for each C3D8 element.
    각 element 별로 1개, 혹은 적분점 별로 1개를 뽑을 수 있다. single=True면 적분점별로 1개

    Input:
        coords (torch.Tensor): Node coordinates [N, 3].
        elements (torch.Tensor): Element connectivity [M, 8].
        E (float): Young's modulus.
        nu (float): Poisson's ratio.
        integral point (torch.Tensor): natural coordinates [xi, eta, zeta, weight], [n_int, 4]
        single (bool)

    Output:
        K_elements (torch.Tensor): Stiffness matrices for all elements [M, 24, 24] / [M, n_int, 24, 24]
    """
    M = elements.shape[0]
    coords = coords.to(device=device, dtype=dtype)
    elements = elements.to(device)
    if single:
        K_elements = torch.zeros((M, 24, 24), device=device, dtype=dtype)
    else:
        K_elements = []

    if integral_point == None:
        integration_points, weights = c3d8_integration_points(device=device, dtype=dtype)  # [8, 3], [8]
    else:
        integration_points, weights = integral_point[:,:3].to(device=device, dtype=dtype), integral_point[:,-1].to(device=device, dtype=dtype)

    D = compute_elasticity_matrix(E, nu, device=device, dtype=dtype)  # [6, 6]

    for ip_idx in range(integration_points.shape[0]):
        ip = integration_points[ip_idx]
        weight = weights[ip_idx]

        B = compute_c3d8_B_matrix(coords, elements, ip, device=device, dtype=dtype)  # [M, 6, 24]
        J = compute_c3d8_Jacobian(coords, elements, ip, device=device, dtype=dtype)  # [M, 3, 3]
        detJ = torch.det(J)  # [M]

        BD = torch.einsum('mji,kj->mki', B, D)  # [M, 6, 24]
        K_ip = torch.einsum('mji,mjk->mik', B, BD)  # [M, 24, 24]
        if single:
            K_elements += K_ip * detJ.view(M, 1, 1) * weight
        else:
            K_elements.append(K_ip*detJ.view(M, 1, 1))

    if single:
        return K_elements  # [M, 24, 24]
    else:
        return torch.stack(K_elements, dim=0)  # [n_int, M, 24, 24]


##### c3d20 노드순서 및 shape function (quadratic)
'''
Node 0:  (-1, -1, -1)
Node 1:  ( 1, -1, -1)
Node 2:  ( 1,  1, -1)
Node 3:  (-1,  1, -1)
Node 4:  (-1, -1,  1)
Node 5:  ( 1, -1,  1)
Node 6:  ( 1,  1,  1)
Node 7:  (-1,  1,  1)
Node 8:  between nodes 0 and 1
Node 9:  between nodes 1 and 2
Node 10: between nodes 2 and 3
Node 11: between nodes 3 and 0
Node 12: between nodes 0 and 4
Node 13: between nodes 1 and 5
Node 14: between nodes 2 and 6
Node 15: between nodes 3 and 7
Node 16: between nodes 4 and 5
Node 17: between nodes 5 and 6
Node 18: between nodes 6 and 7
Node 19: between nodes 7 and 4

1. Corner Nodes (0-7):
N_0 = (1 - ξ)(1 - η)(1 - ζ)(-2 - ξ - η - ζ)/8
N_1 = (1 + ξ)(1 - η)(1 - ζ)( 2 + ξ - η - ζ)/8
N_2 = (1 + ξ)(1 + η)(1 - ζ)( 2 + ξ + η - ζ)/8
N_3 = (1 - ξ)(1 + η)(1 - ζ)(-2 + ξ + η - ζ)/8
N_4 = (1 - ξ)(1 - η)(1 + ζ)(-2 - ξ - η + ζ)/8
N_5 = (1 + ξ)(1 - η)(1 + ζ)( 2 + ξ - η + ζ)/8
N_6 = (1 + ξ)(1 + η)(1 + ζ)( 2 + ξ + η + ζ)/8
N_7 = (1 - ξ)(1 + η)(1 + ζ)(-2 + ξ + η + ζ)/8
2. Mid-Side Nodes (8-19):
N_8  = (1 - ξ²)(1 - η)(1 - ζ)/4
N_9  = (1 - ξ²)(1 + η)(1 - ζ)/4
N_10 = (1 - ξ²)(1 + η)(1 + ζ)/4
N_11 = (1 - ξ²)(1 - η)(1 + ζ)/4
N_12 = (1 - ξ)(1 - η²)(1 - ζ)/4
N_13 = (1 + ξ)(1 - η²)(1 - ζ)/4
N_14 = (1 + ξ)(1 - η²)(1 + ζ)/4
N_15 = (1 - ξ)(1 - η²)(1 + ζ)/4
N_16 = (1 - ξ)(1 - η)(1 - ζ²)/4
N_17 = (1 + ξ)(1 - η)(1 - ζ²)/4
N_18 = (1 + ξ)(1 + η)(1 - ζ²)/4
N_19 = (1 - ξ)(1 + η)(1 - ζ²)/4
'''
def c3d20_to_c3d4(c3d20_elements, device="cuda:0"):
    """
    Fully separate C3D20 elements into 6 C3D4 elements to cover the entire volume.

    Input:
        c3d20_elements (torch.Tensor): Tensor of C3D8 elements [M, 20]

    Output:
        c3d4_elements (torch.Tensor): C3D4 elements [6*M, 4]
    """
    c3d20_elements = c3d20_elements.to(device)

    sub_tetrahedra = torch.tensor([
        [0, 8, 12, 19],
        [8, 1, 13, 9],
        [9, 1, 2, 10],
        [10, 2, 14, 11],
        [11, 2, 3, 15],
        [15, 3, 19, 0],
        [12, 4, 16, 19],
        [16, 4, 5, 17],
        [17, 5, 13, 18],
        [18, 5, 6, 14],
        [14, 6, 18, 7],
        [19, 7, 15, 11],
        [8, 9, 10, 11],
        [8, 10, 11, 12],
        [12, 13, 14, 15],
        [16, 17, 18, 19],
        [0, 8, 9, 10],
        [0, 10, 11, 12],
        [1, 9, 10, 13],
        [1, 13, 14, 17],
        [2, 10, 14, 15],
        [3, 11, 15, 19],
        [4, 12, 16, 19],
        [5, 13, 17, 18]
    ], dtype=torch.long, device=device)  # [6, 4]

    M = c3d20_elements.shape[0]

    c3d4_elements = c3d20_elements[:, sub_tetrahedra]  # [M, 24, 4]
    c3d4_elements = c3d4_elements.view(-1, 4)  # [24*M, 4]

    return c3d4_elements

def c3d20_integration_points(device="cuda:0", dtype=torch.float32):
    sqrt_3_5 = torch.sqrt(torch.tensor(3.0 / 5.0))

    gauss_points_1D = torch.tensor([-sqrt_3_5, 0.0, sqrt_3_5], dtype=dtype)
    gauss_weights_1D = torch.tensor([5.0/9.0, 8.0/9.0, 5.0/9.0], dtype=dtype)

    c3d20_integration_points = torch.zeros((27, 3), dtype=dtype)
    c3d20_weights = torch.zeros(27, dtype=dtype)

    index = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                r = gauss_points_1D[i]
                s = gauss_points_1D[j]
                t = gauss_points_1D[k]
                w = gauss_weights_1D[i] * gauss_weights_1D[j] * gauss_weights_1D[k]
                c3d20_integration_points[index] = torch.tensor([r, s, t], dtype=dtype)
                c3d20_weights[index] = w
                index += 1
    
    return c3d20_integration_points.to(device), c3d20_weights.to(device)

def compute_c3d20_Jacobian(coords, elements, integral_point, device="cuda:0", dtype=torch.float32):
    """
    주어진 적분점에서 각 c3d20 육면체에 대해 야코비안을 구하는 함수

    Input:
        coords (torch.Tensor): Node coordinates [N, 3]
        elements (torch.Tensor): Element connectivity [M, 20]
        integral_point (torch.Tensor): Natural coordinates [xi, eta, zeta]

    Output:
        Jacobian (torch.Tensor): Jacobian matrices [M, 3, 3]
    """
    coords = coords.to(device=device, dtype=dtype)
    elements = elements.to(device)
    xi, eta, zeta = integral_point.to(device=device, dtype=dtype)

    dN_dnatural = torch.tensor([
        # Corner Nodes (0–7)
        [-0.125 * (1 - eta) * (1 - zeta) * (-2 - xi - eta - zeta) + 0.125 * (1 - eta) * (1 - zeta),  # Node 0
         -0.125 * (1 - xi) * (1 - zeta) * (-2 - xi - eta - zeta) + 0.125 * (1 - xi) * (1 - zeta),
         -0.125 * (1 - xi) * (1 - eta) * (-2 - xi - eta - zeta) + 0.125 * (1 - xi) * (1 - eta)],
        [0.125 * (1 - eta) * (1 - zeta) * (-2 + xi - eta - zeta) + 0.125 * (1 - eta) * (1 - zeta),  # Node 1
         -0.125 * (1 + xi) * (1 - zeta) * (-2 + xi - eta - zeta) + 0.125 * (1 + xi) * (1 - zeta),
         -0.125 * (1 + xi) * (1 - eta) * (-2 + xi - eta - zeta) + 0.125 * (1 + xi) * (1 - eta)],
        [0.125 * (1 + eta) * (1 - zeta) * (-2 + xi + eta - zeta) + 0.125 * (1 + eta) * (1 - zeta),  # Node 2
         0.125 * (1 + xi) * (1 - zeta) * (-2 + xi + eta - zeta) + 0.125 * (1 + xi) * (1 - zeta),
         -0.125 * (1 + xi) * (1 + eta) * (-2 + xi + eta - zeta) + 0.125 * (1 + xi) * (1 + eta)],
        [-0.125 * (1 + eta) * (1 - zeta) * (-2 - xi + eta - zeta) + 0.125 * (1 + eta) * (1 - zeta),  # Node 3
         0.125 * (1 - xi) * (1 - zeta) * (-2 - xi + eta - zeta) + 0.125 * (1 - xi) * (1 - zeta),
         -0.125 * (1 - xi) * (1 + eta) * (-2 - xi + eta - zeta) + 0.125 * (1 - xi) * (1 + eta)],
        [-0.125 * (1 - eta) * (1 + zeta) * (-2 - xi - eta + zeta) + 0.125 * (1 - eta) * (1 + zeta),  # Node 4
         -0.125 * (1 - xi) * (1 + zeta) * (-2 - xi - eta + zeta) + 0.125 * (1 - xi) * (1 + zeta),
         0.125 * (1 - xi) * (1 - eta) * (-2 - xi - eta + zeta) + 0.125 * (1 - xi) * (1 - eta)],
        [0.125 * (1 - eta) * (1 + zeta) * (-2 + xi - eta + zeta) + 0.125 * (1 - eta) * (1 + zeta),  # Node 5
         -0.125 * (1 + xi) * (1 + zeta) * (-2 + xi - eta + zeta) + 0.125 * (1 + xi) * (1 + zeta),
         0.125 * (1 + xi) * (1 - eta) * (-2 + xi - eta + zeta) + 0.125 * (1 + xi) * (1 - eta)],
        [0.125 * (1 + eta) * (1 + zeta) * (-2 + xi + eta + zeta) + 0.125 * (1 + eta) * (1 + zeta),  # Node 6
         0.125 * (1 + xi) * (1 + zeta) * (-2 + xi + eta + zeta) + 0.125 * (1 + xi) * (1 + zeta),
         0.125 * (1 + xi) * (1 + eta) * (-2 + xi + eta + zeta) + 0.125 * (1 + xi) * (1 + eta)],
        [-0.125 * (1 + eta) * (1 + zeta) * (-2 - xi + eta + zeta) + 0.125 * (1 + eta) * (1 + zeta),  # Node 7
         0.125 * (1 - xi) * (1 + zeta) * (-2 - xi + eta + zeta) + 0.125 * (1 - xi) * (1 + zeta),
         0.125 * (1 - xi) * (1 + eta) * (-2 - xi + eta + zeta) + 0.125 * (1 - xi) * (1 + eta)],
        # Mid-Side Nodes (8–19)
        [-0.25 * xi * (1 - eta) * (1 - zeta), -0.25 * (1 - xi**2) * (1 - zeta), -0.25 * (1 - xi**2) * (1 - eta)],  # Node 8
        [0.25 * (1 - eta**2) * (1 - zeta), -0.25 * eta * (1 + xi) * (1 - zeta), -0.25 * (1 - eta**2) * (1 + xi)],  # Node 9
        [-0.25 * xi * (1 + eta) * (1 - zeta), 0.25 * (1 - xi**2) * (1 - zeta), -0.25 * (1 - xi**2) * (1 + eta)],   # Node 10
        [-0.25 * (1 - eta**2) * (1 - zeta), 0.25 * eta * (1 - xi) * (1 - zeta), -0.25 * (1 - eta**2) * (1 - xi)],  # Node 11
        [-0.25 * xi * (1 - eta) * (1 + zeta), -0.25 * (1 - xi**2) * (1 + zeta), 0.25 * (1 - xi**2) * (1 - eta)],   # Node 12
        [0.25 * (1 - eta**2) * (1 + zeta), -0.25 * eta * (1 + xi) * (1 + zeta), 0.25 * (1 - eta**2) * (1 + xi)],   # Node 13
        [-0.25 * xi * (1 + eta) * (1 + zeta), 0.25 * (1 - xi**2) * (1 + zeta), 0.25 * (1 - xi**2) * (1 + eta)],    # Node 14
        [-0.25 * (1 - eta**2) * (1 + zeta), 0.25 * eta * (1 - xi) * (1 + zeta), 0.25 * (1 - eta**2) * (1 - xi)],   # Node 15
        [-0.25 * xi * (1 - eta) * (1 - zeta**2), -0.25 * (1 - xi**2) * (1 - zeta**2), 0.25 * (1 - xi**2) * (1 - eta)],  # Node 16
        [0.25 * (1 - eta**2) * (1 - zeta**2), -0.25 * eta * (1 + xi) * (1 - zeta**2), 0.25 * (1 - eta**2) * (1 + xi)],  # Node 17
        [-0.25 * xi * (1 + eta) * (1 - zeta**2), 0.25 * (1 - xi**2) * (1 - zeta**2), 0.25 * (1 - xi**2) * (1 + eta)],   # Node 18
        [-0.25 * (1 - eta**2) * (1 - zeta**2), 0.25 * eta * (1 - xi) * (1 - zeta**2), 0.25 * (1 - eta**2) * (1 - xi)]    # Node 19
    ]).to(device=device, dtype=dtype)  # (20, 3)

    element_coords = coords[elements]  # (M, 20, 3)

    jacobian = torch.einsum("ij,mjk->mik", dN_dnatural, element_coords)  # (M, 3, 3)

    return jacobian

def compute_c3d20_shape_gradients(coords, elements, integral_point, device="cuda:0", dtype=torch.float32):
    """
    주어진 적분점에서 각 c3d20 육면체에 대해 shape function gradient를 구하는 함수

    Input:
        coords (torch.Tensor): Node coordinates [N, 3]
        elements (torch.Tensor): Element connectivity [M, 20]
        integral_point (torch.Tensor): Natural coordinates [xi, eta, zeta]

    Output:
        dN_global (torch.Tensor): Shape function gradients in global coordinates [M, 20, 3]
    """
    jacobian = compute_c3d20_Jacobian(coords, elements, integral_point, device=device, dtype=dtype)  # (M, 3, 3)

    jacobian_inv = torch.inverse(jacobian)  # (M, 3, 3)

    xi, eta, zeta = integral_point.to(device=device, dtype=dtype)
    dN_dnatural = torch.tensor([
        # Corner Nodes (0–7)
        [-0.125 * (1 - eta) * (1 - zeta) * (-2 - xi - eta - zeta) + 0.125 * (1 - eta) * (1 - zeta),  # Node 0
         -0.125 * (1 - xi) * (1 - zeta) * (-2 - xi - eta - zeta) + 0.125 * (1 - xi) * (1 - zeta),
         -0.125 * (1 - xi) * (1 - eta) * (-2 - xi - eta - zeta) + 0.125 * (1 - xi) * (1 - eta)],
        [0.125 * (1 - eta) * (1 - zeta) * (-2 + xi - eta - zeta) + 0.125 * (1 - eta) * (1 - zeta),  # Node 1
         -0.125 * (1 + xi) * (1 - zeta) * (-2 + xi - eta - zeta) + 0.125 * (1 + xi) * (1 - zeta),
         -0.125 * (1 + xi) * (1 - eta) * (-2 + xi - eta - zeta) + 0.125 * (1 + xi) * (1 - eta)],
        [0.125 * (1 + eta) * (1 - zeta) * (-2 + xi + eta - zeta) + 0.125 * (1 + eta) * (1 - zeta),  # Node 2
         0.125 * (1 + xi) * (1 - zeta) * (-2 + xi + eta - zeta) + 0.125 * (1 + xi) * (1 - zeta),
         -0.125 * (1 + xi) * (1 + eta) * (-2 + xi + eta - zeta) + 0.125 * (1 + xi) * (1 + eta)],
        [-0.125 * (1 + eta) * (1 - zeta) * (-2 - xi + eta - zeta) + 0.125 * (1 + eta) * (1 - zeta),  # Node 3
         0.125 * (1 - xi) * (1 - zeta) * (-2 - xi + eta - zeta) + 0.125 * (1 - xi) * (1 - zeta),
         -0.125 * (1 - xi) * (1 + eta) * (-2 - xi + eta - zeta) + 0.125 * (1 - xi) * (1 + eta)],
        [-0.125 * (1 - eta) * (1 + zeta) * (-2 - xi - eta + zeta) + 0.125 * (1 - eta) * (1 + zeta),  # Node 4
         -0.125 * (1 - xi) * (1 + zeta) * (-2 - xi - eta + zeta) + 0.125 * (1 - xi) * (1 + zeta),
         0.125 * (1 - xi) * (1 - eta) * (-2 - xi - eta + zeta) + 0.125 * (1 - xi) * (1 - eta)],
        [0.125 * (1 - eta) * (1 + zeta) * (-2 + xi - eta + zeta) + 0.125 * (1 - eta) * (1 + zeta),  # Node 5
         -0.125 * (1 + xi) * (1 + zeta) * (-2 + xi - eta + zeta) + 0.125 * (1 + xi) * (1 + zeta),
         0.125 * (1 + xi) * (1 - eta) * (-2 + xi - eta + zeta) + 0.125 * (1 + xi) * (1 - eta)],
        [0.125 * (1 + eta) * (1 + zeta) * (-2 + xi + eta + zeta) + 0.125 * (1 + eta) * (1 + zeta),  # Node 6
         0.125 * (1 + xi) * (1 + zeta) * (-2 + xi + eta + zeta) + 0.125 * (1 + xi) * (1 + zeta),
         0.125 * (1 + xi) * (1 + eta) * (-2 + xi + eta + zeta) + 0.125 * (1 + xi) * (1 + eta)],
        [-0.125 * (1 + eta) * (1 + zeta) * (-2 - xi + eta + zeta) + 0.125 * (1 + eta) * (1 + zeta),  # Node 7
         0.125 * (1 - xi) * (1 + zeta) * (-2 - xi + eta + zeta) + 0.125 * (1 - xi) * (1 + zeta),
         0.125 * (1 - xi) * (1 + eta) * (-2 - xi + eta + zeta) + 0.125 * (1 - xi) * (1 + eta)],
        # Mid-Side Nodes (8–19)
        [-0.25 * xi * (1 - eta) * (1 - zeta), -0.25 * (1 - xi**2) * (1 - zeta), -0.25 * (1 - xi**2) * (1 - eta)],  # Node 8
        [0.25 * (1 - eta**2) * (1 - zeta), -0.25 * eta * (1 + xi) * (1 - zeta), -0.25 * (1 - eta**2) * (1 + xi)],  # Node 9
        [-0.25 * xi * (1 + eta) * (1 - zeta), 0.25 * (1 - xi**2) * (1 - zeta), -0.25 * (1 - xi**2) * (1 + eta)],   # Node 10
        [-0.25 * (1 - eta**2) * (1 - zeta), 0.25 * eta * (1 - xi) * (1 - zeta), -0.25 * (1 - eta**2) * (1 - xi)],  # Node 11
        [-0.25 * xi * (1 - eta) * (1 + zeta), -0.25 * (1 - xi**2) * (1 + zeta), 0.25 * (1 - xi**2) * (1 - eta)],   # Node 12
        [0.25 * (1 - eta**2) * (1 + zeta), -0.25 * eta * (1 + xi) * (1 + zeta), 0.25 * (1 - eta**2) * (1 + xi)],   # Node 13
        [-0.25 * xi * (1 + eta) * (1 + zeta), 0.25 * (1 - xi**2) * (1 + zeta), 0.25 * (1 - xi**2) * (1 + eta)],    # Node 14
        [-0.25 * (1 - eta**2) * (1 + zeta), 0.25 * eta * (1 - xi) * (1 + zeta), 0.25 * (1 - eta**2) * (1 - xi)],   # Node 15
        [-0.25 * xi * (1 - eta) * (1 - zeta**2), -0.25 * (1 - xi**2) * (1 - zeta**2), 0.25 * (1 - xi**2) * (1 - eta)],  # Node 16
        [0.25 * (1 - eta**2) * (1 - zeta**2), -0.25 * eta * (1 + xi) * (1 - zeta**2), 0.25 * (1 - eta**2) * (1 + xi)],  # Node 17
        [-0.25 * xi * (1 + eta) * (1 - zeta**2), 0.25 * (1 - xi**2) * (1 - zeta**2), 0.25 * (1 - xi**2) * (1 + eta)],   # Node 18
        [-0.25 * (1 - eta**2) * (1 - zeta**2), 0.25 * eta * (1 - xi) * (1 - zeta**2), 0.25 * (1 - eta**2) * (1 - xi)]    # Node 19
    ]).to(device=device, dtype=dtype)  # (20, 3)

    dN_global = torch.einsum("mij,nj->mni", jacobian_inv, dN_dnatural)  # (M, 20, 3)

    return dN_global

def compute_c3d20_B_matrix(coords, elements, integral_point, device="cuda:0", dtype=torch.float32):
    """
    주어진 적분점에서 각 c3d20 육면체에 대해 B 행렬을 구하는 함수

    Input:
        coords (torch.Tensor): Node coordinates [N, 3]
        elements (torch.Tensor): Element connectivity [M, 20]
        integral_point (torch.Tensor): Natural coordinates [xi, eta, zeta]

    Output:
        B_matrices (torch.Tensor): B matrices for all elements [M, 6, 60]
    """
    dN_global = compute_c3d20_shape_gradients(coords, elements, integral_point, device=device, dtype=dtype)  # [M, 20, 3]

    M = elements.shape[0]  
    B_matrices = torch.zeros((M, 6, 60), device=device, dtype=dtype)  # [M, 6, 60]

    for i in range(20): 
        B_matrices[:, 0, 3 * i + 0] = dN_global[:, i, 0]  # ∂N/∂x
        B_matrices[:, 1, 3 * i + 1] = dN_global[:, i, 1]  # ∂N/∂y
        B_matrices[:, 2, 3 * i + 2] = dN_global[:, i, 2]  # ∂N/∂z
        B_matrices[:, 3, 3 * i + 0] = dN_global[:, i, 1]  # ∂N/∂y
        B_matrices[:, 3, 3 * i + 1] = dN_global[:, i, 0]  # ∂N/∂x
        B_matrices[:, 4, 3 * i + 1] = dN_global[:, i, 2]  # ∂N/∂z
        B_matrices[:, 4, 3 * i + 2] = dN_global[:, i, 1]  # ∂N/∂y
        B_matrices[:, 5, 3 * i + 0] = dN_global[:, i, 2]  # ∂N/∂z
        B_matrices[:, 5, 3 * i + 2] = dN_global[:, i, 0]  # ∂N/∂x

    return B_matrices

def compute_c3d20_element_stress(coords, elements, displacement, E, nu, integral_point=None, single=True, device="cuda:0", dtype=torch.float32):
    """
    최종적으로, 좌표 정보, 육면체 정보, 변위 정보가 주어지면 element단위의 스트레스를 적분점별로 계산하는 함수.
    Young's Modulus와 Poisson Ratio 또한 주어져야한다.
    적분점도 주어지며, [n_int, 4] 형태이다. None이면 기본 11점 integration point를 사용한다.
    각 element 별로 1개, 혹은 적분점 별로 1개를 뽑을 수 있다. single=True면 적분점별로 1개

    Input:
        coords (torch.Tensor): Node coordinates [N, 3]
        elements (torch.Tensor): Element connectivity [M, 20]
        displacement (torch.Tensor): Node Displacements [N, 3]
        E (float): Young's modulus
        nu (float): Poisson's ratio
        integral point (torch.Tensor): natural coordinates [xi, eta, zeta, weight], [n_int, 4]


    Output:
        tuple:
            stress_per_ip (torch.Tensor): Ip-Element-wise stress tensors [M,n_int,3,3]
            vm_stress_per_ip (torch.Tensor): Ip-Element-wise von Mises stress [M,n_int]
    """
    M = elements.shape[0]
    coords = coords.to(device=device, dtype=dtype)
    elements = elements.to(device)
    displacement = displacement.to(device=device, dtype=dtype)

    if integral_point == None:
        p, w = c3d20_integration_points(device=device, dtype=dtype)
    else:
        p, w = integral_point[:,:3].to(device=device, dtype=dtype), integral_point[:,-1].to(device=device, dtype=dtype)
    
    D = compute_elasticity_matrix(E, nu, device=device, dtype=dtype)
    disp_elem = displacement[elements].reshape(M, -1)  # [M, 60]

    if single:
        stress_result = torch.zeros((M, 3, 3), device=device, dtype=dtype)
        vm_stress_result = torch.zeros((M), device=device, dtype=dtype)
    else:
        stress_result = []
        vm_stress_result = []

    for idx in range(p.shape[0]):
        ip = p[idx]
        
        B = compute_c3d20_B_matrix(coords, elements, ip, device=device) # [M, 6, 60]
        
        strain = torch.bmm(B, disp_elem.unsqueeze(2)).squeeze(2)  # [M,6]
        stress = torch.matmul(strain, D.t())  # [M,6]

        stress_tensor = compute_stress_tensor(stress)  # [M,3,3]
        element_vm_stress = compute_von_mises_stress(stress_tensor)  # [M]

        if single:
            stress_result += stress_tensor * w[idx]
            vm_stress_result += element_vm_stress * w[idx]
        else:
            stress_result.append(stress_tensor)
            vm_stress_result.append(element_vm_stress)

    if single:
        return stress_result, vm_stress_result # [M,3,3], [M]
    else:
        return torch.tensor(np.array(stress_result), device=device, dtype=dtype), torch.tensor(np.array(vm_stress_result), device=device, dtype=dtype) # [n_int, M, 3, 3], [n_int, M]

def compute_c3d20_K_matrix(coords, elements, E, nu, integral_point=None, single=True, device="cuda:0", dtype=torch.float32):
    """
    Computes the element stiffness matrix for each C3D20 element.
    각 element 별로 1개, 혹은 적분점 별로 1개를 뽑을 수 있다. single=True면 적분점별로 1개

    Input:
        coords (torch.Tensor): Node coordinates [N, 3].
        elements (torch.Tensor): Element connectivity [M, 20].
        E (float): Young's modulus.
        nu (float): Poisson's ratio.
        integral point (torch.Tensor): natural coordinates [xi, eta, zeta, weight], [n_int, 4]
        single (bool)

    Output:
        K_elements (torch.Tensor): Stiffness matrices for all elements [M, 60, 60] / [M, n_int, 60, 60]
    """
    M = elements.shape[0]
    coords = coords.to(device=device, dtype=dtype)
    elements = elements.to(device)
    if single:
        K_elements = torch.zeros((M, 60, 60), device=device, dtype=dtype)
    else:
        K_elements = []

    if integral_point == None:
        integration_points, weights = c3d20_integration_points(device=device, dtype=dtype)  # [27, 3], [27]
    else:
        integration_points, weights = integral_point[:,:3].to(device=device, dtype=dtype), integral_point[:,-1].to(device=device, dtype=dtype)

    D = compute_elasticity_matrix(E, nu, device=device, dtype=dtype)  # [6, 6]

    for ip_idx in range(integration_points.shape[0]):
        ip = integration_points[ip_idx]
        weight = weights[ip_idx]

        B = compute_c3d20_B_matrix(coords, elements, ip, device=device, dtype=dtype)  # [M, 6, 60]
        J = compute_c3d20_Jacobian(coords, elements, ip, device=device, dtype=dtype)  # [M, 3, 3]
        detJ = torch.det(J)  # [M]

        K_ip = torch.einsum('mia,ij,mjb->mab', B, D, B) # [M, 60, 60]
        if single:
            K_elements += K_ip * detJ.view(M, 1, 1) * weight
        else:
            K_elements.append(K_ip*detJ.view(M, 1, 1))

    if single:
        return K_elements  # [M, 60, 60]
    else:
        return torch.tensor(np.array(K_elements), device=device, dtype=dtype)  # [n_int, M, 60, 60]



#################################################################################################
###########################################   Wedge   ###########################################
#################################################################################################


##### 기초연산 (c3d6, c3d15 모두 사용)
def compute_wedge_volumes(coords, elements, device="cuda:0", dtype=torch.float32):
    """
    Computes the volumes of wedge elements (C3D6).

    Args:
        coords (torch.Tensor): Node coordinates [N, 3].
        elements (torch.Tensor): Element connectivity [M, 6].

    Returns:
        volumes (torch.Tensor): Volumes of wedge elements [M].
    """
    coords = coords.to(device=device, dtype=dtype)
    elements = elements.to(device)

    p0 = coords[elements[:, 0]]  # Node 0
    p1 = coords[elements[:, 1]]  # Node 1
    p2 = coords[elements[:, 2]]  # Node 2
    p3 = coords[elements[:, 3]]  # Node 3
    p4 = coords[elements[:, 4]]  # Node 4
    p5 = coords[elements[:, 5]]  # Node 5

    def volume_tetrahedron(a, b, c, d):
        """
        Computes the volume of a tetrahedron given its vertices.
        """
        v = torch.abs(torch.einsum('ij,ij->i', torch.cross(b - a, c - a), d - a)) / 6.0
        return v

    vol1 = volume_tetrahedron(p0, p1, p2, p3)
    vol2 = volume_tetrahedron(p1, p2, p4, p3)
    vol3 = volume_tetrahedron(p2, p4, p5, p3)

    volumes = vol1 + vol2 + vol3

    return volumes # [M]

def compute_wedge_surface_faces_with_extra_node(elements, device="cuda:0"):
    """
    Identifies surface faces and extra nodes for wedge elements (C3D6).

    Args:
        elements (torch.Tensor): Element connectivity [M, 6].

    Returns:
        surface_faces (list of torch.Tensor): Surface faces [K_quad, 4], [K_tri, 3].
        extra_nodes (list of torch.Tensor): Extra nodes for each face [K_quad], [K_tri].
    """
    elements = elements.to(device)
    M = elements.shape[0]
    
    quad_faces = torch.cat([
        elements[:, [0, 1, 4, 3]],  # Side face 1
        elements[:, [1, 2, 5, 4]],  # Side face 2
        elements[:, [2, 0, 3, 5]],  # Side face 3
    ], dim=0)  # [M*3, 4]
    quad_extra_nodes = torch.cat([
        elements[:, 2],  # Opposite node for face 1
        elements[:, 0],  # Opposite node for face 2
        elements[:, 1],  # Opposite node for face 3
    ], dim=0)

    tri_faces = torch.cat([
        elements[:, [0, 2, 1]],     # Bottom face
        elements[:, [3, 4, 5]],     # Top face
    ], dim=0)  # [M*2, 3]
    tri_extra_nodes = torch.cat([
        elements[:, 3],  # Opposite node for bottom face
        elements[:, 0],  # Opposite node for top face
    ], dim=0)

    sorted_quad_faces, _ = torch.sort(quad_faces, dim=1)
    unique_quad_faces, inverse_indices_quad, counts_quad = torch.unique(sorted_quad_faces, dim=0, return_inverse=True, return_counts=True)
    surface_face_mask_quad = counts_quad[inverse_indices_quad] == 1
    surface_quad_faces = quad_faces[surface_face_mask_quad]
    surface_quad_extra_nodes = quad_extra_nodes[surface_face_mask_quad]

    sorted_tri_faces, _ = torch.sort(tri_faces, dim=1)
    unique_tri_faces, inverse_indices_tri, counts_tri = torch.unique(sorted_tri_faces, dim=0, return_inverse=True, return_counts=True)
    surface_face_mask_tri = counts_tri[inverse_indices_tri] == 1
    surface_tri_faces = tri_faces[surface_face_mask_tri]
    surface_tri_extra_nodes = tri_extra_nodes[surface_face_mask_tri]

    surface_faces = [surface_quad_faces, surface_tri_faces]
    extra_nodes = [surface_quad_extra_nodes, surface_tri_extra_nodes]

    return surface_faces, extra_nodes

def compute_wedge_surface_normals(coords, elements, device="cuda:0", dtype=torch.float32):
    """
    Computes outward normals for surface faces of wedge elements.

    Args:
        coords (torch.Tensor): Node coordinates [N, 3].
        elements (torch.Tensor): Element connectivity [M, 6].

    Returns:
        surface_normals (list of torch.Tensor): Normals for each face [K_quad, 3], [K_tri, 3].
    """
    coords = coords.to(device=device, dtype=dtype)
    elements = elements.to(device)

    surface_faces, extra_nodes = compute_wedge_surface_faces_with_extra_node(elements, device=device)
    surface_normals = []

    for faces, extra_nodes_per_face in zip(surface_faces, extra_nodes):
        faces = faces.to(device)
        extra_nodes_per_face = extra_nodes_per_face.to(device)
        n_nodes_per_face = faces.shape[1]

        if n_nodes_per_face == 4:  # Quadrilateral faces
            p1 = coords[faces[:, 0]]
            p2 = coords[faces[:, 1]]
            p3 = coords[faces[:, 2]]
            p4 = coords[faces[:, 3]]

            v1 = p2 - p1
            v2 = p4 - p1
            normals = torch.cross(v1, v2, dim=1)
        else:  # Triangular faces
            p1 = coords[faces[:, 0]]
            p2 = coords[faces[:, 1]]
            p3 = coords[faces[:, 2]]

            v1 = p2 - p1
            v2 = p3 - p1
            normals = torch.cross(v1, v2, dim=1)

        normals = normals / torch.norm(normals, dim=1, keepdim=True)

        face_centers = coords[faces].mean(dim=1)
        extra_points = coords[extra_nodes_per_face]

        to_extra = extra_points - face_centers
        to_extra_normalized = to_extra / torch.norm(to_extra, dim=1, keepdim=True)

        dot_product = (normals * to_extra_normalized).sum(dim=1)
        normals[dot_product > 0] = -normals[dot_product > 0]

        surface_normals.append(normals)

    return surface_normals # [K_quad, 3], [K_tri, 3]

def compute_wedge_node_curvatures(coords, elements, device="cuda:0", dtype=torch.float32):
    """
    Computes mean curvature vectors at nodes for wedge elements.

    Args:
        coords (torch.Tensor): Node coordinates [N, 3].
        elements (torch.Tensor): Element connectivity [M, 6].

    Returns:
        mean_curvature_vectors (torch.Tensor): Mean curvature vectors at nodes [N, 3].
    """
    coords = coords.to(device=device, dtype=dtype)
    elements = elements.to(device)

    surface_faces, face_normals = compute_wedge_surface_normals(coords, elements, device=device, dtype=dtype)
    N = coords.shape[0]

    accumulated_curvatures = torch.zeros((N, 3), device=device)
    node_counts = torch.zeros(N, device=device)

    for faces, normals in zip(surface_faces, face_normals):
        faces = faces.to(device)
        normals = normals.to(device)
        n_nodes_per_face = faces.shape[1]
        normals_expanded = normals[:, None, :].expand(-1, n_nodes_per_face, -1)
        accumulated_curvatures.scatter_add_(
            0,
            faces.view(-1, 1).expand(-1, 3),
            normals_expanded.reshape(-1, 3)
        )
        ones = torch.ones(faces.numel(), device=device)
        node_counts.scatter_add_(0, faces.view(-1), ones)

    node_counts[node_counts == 0] = 1
    mean_curvature_vectors = accumulated_curvatures / node_counts.view(-1, 1)
    return mean_curvature_vectors # [N, 3]

def compute_wedge_normals_and_area(coords, elements, device="cuda:0", dtype=torch.float32):
    """
    Computes normals and areas for each face of wedge elements.

    Args:
        coords (torch.Tensor): Node coordinates [N, 3].
        elements (torch.Tensor): Element connectivity [M, 6].

    Returns:
        normal_vectors (torch.Tensor): Normals for each face [M, 5, 3].
    """
    coords = coords.to(device=device, dtype=dtype)
    elements = elements.to(device)
    M = elements.shape[0]

    # Define face node indices
    quad_faces_indices = torch.tensor([
        [0, 1, 4, 3],
        [1, 2, 5, 4],
        [2, 0, 3, 5]
    ], device=device)
    tri_faces_indices = torch.tensor([
        [0, 2, 1],
        [3, 4, 5]
    ], device=device)

    coords_elem = coords[elements]

    # Quadrilateral faces
    quad_face_coords = coords_elem[:, quad_faces_indices]
    edge1_quad = quad_face_coords[:, :, 1, :] - quad_face_coords[:, :, 0, :]
    edge2_quad = quad_face_coords[:, :, 3, :] - quad_face_coords[:, :, 0, :]
    normals_quad = torch.cross(edge1_quad, edge2_quad, dim=3)

    # Triangular faces
    tri_face_coords = coords_elem[:, tri_faces_indices]
    edge1_tri = tri_face_coords[:, :, 1, :] - tri_face_coords[:, :, 0, :]
    edge2_tri = tri_face_coords[:, :, 2, :] - tri_face_coords[:, :, 0, :]
    normals_tri = torch.cross(edge1_tri, edge2_tri, dim=3)

    normals = torch.cat([normals_quad, normals_tri], dim=1)
    normals = normals / torch.norm(normals, dim=3, keepdim=True)

    return normals  # [M, 5, 3]


##### c3d6
def c3d6_to_c3d4(element, device="cuda:0"):
    """
    Converts a C3D6 element to 4 C3D4 elements.

    Args:
        element (torch.Tensor): Element connectivity [6].

    Returns:
        c3d4_elements (torch.Tensor): Element connectivity [4, 4].
    """

    sub_tetrahedra = torch.tensor([
        [0, 1, 2, 3],  # Tetrahedron 1
        [1, 2, 3, 5],  # Tetrahedron 2
        [1, 3, 4, 5],  # Tetrahedron 3
    ], dtype=torch.long, device=device) # [3, 4]

    M = element.shape[0]

    c3d4_elements = element[:, sub_tetrahedra] # [M, 3, 4]
    c3d4_elements = c3d4_elements.view(-1, 4) # [M*3, 4]

    return c3d4_elements

def c3d6_integration_points(device="cuda:0", dtype=torch.float32):
    """
    Returns the integration points and weights for the C3D6 element.
    We'll use a 2x2 Gauss quadrature for the triangle and 2-point Gauss quadrature along the length.
    """
    triangle_points = torch.tensor([
        [1/6, 1/6],
        [2/3, 1/6],
        [1/6, 2/3]
    ], dtype=dtype, device=device)
    triangle_weights = torch.tensor([1/3, 1/3, 1/3], dtype=dtype, device=device)
    
    line_points = torch.tensor([
        -1.0 / torch.sqrt(torch.tensor(3.0)),
         1.0 / torch.sqrt(torch.tensor(3.0))
    ], dtype=dtype, device=device)
    line_weights = torch.tensor([1.0, 1.0], dtype=dtype, device=device)
    
    integration_points = []
    weights = []
    for i in range(len(triangle_points)):
        for j in range(len(line_points)):
            xi = triangle_points[i][0]
            eta = triangle_points[i][1]
            zeta = line_points[j]
            weight = triangle_weights[i] * line_weights[j]
            integration_points.append([xi, eta, zeta])
            weights.append(weight)
    
    integration_points = torch.tensor(integration_points, dtype=dtype, device=device)
    weights = torch.tensor(weights, dtype=dtype, device=device)
    
    return integration_points, weights

def compute_c3d6_Jacobian(coords, elements, integral_point, device="cuda:0", dtype=torch.float32):
    """
    Computes the Jacobian matrices for the C3D6 element at the given integration point.

    Args:
        coords (torch.Tensor): Node coordinates [N, 3]
        elements (torch.Tensor): Element connectivity [M, 6]
        integral_point (torch.Tensor): Natural coordinates [r, s, t]

    Returns:
        jacobian (torch.Tensor): Jacobian matrices [M, 3, 3]
    """
    r, s, t = integral_point.to(device=device, dtype=dtype)
    coords = coords.to(device=device, dtype=dtype)
    elements = elements.to(device)
    element_coords = coords[elements]  # [M, 6, 3]
    
    dN_dnatural = torch.tensor([
        [ -0.5*(1 - t), -0.5*(1 - t), -0.5*(1 - r - s) ],
        [  0.5*(1 - t),  0.0,        -0.5*r         ],
        [  0.0,         0.5*(1 - t), -0.5*s         ],
        [ -0.5*(1 + t), -0.5*(1 + t),  0.5*(1 - r - s) ],
        [  0.5*(1 + t),  0.0,         0.5*r         ],
        [  0.0,         0.5*(1 + t),  0.5*s         ]
    ], dtype=dtype, device=device)  # [6, 3]
    
    jacobian = torch.einsum('ji,mjk->mik', dN_dnatural, element_coords)  # [M, 3, 3]
    return jacobian

def compute_c3d6_shape_gradients(coords, elements, integral_point, device="cuda:0", dtype=torch.float32):
    """
    Computes the shape function gradients for the C3D6 element at the given integration point.

    Args:
        coords (torch.Tensor): Node coordinates [N, 3]
        elements (torch.Tensor): Element connectivity [M, 6]
        integral_point (torch.Tensor): Natural coordinates [r, s, t]

    Returns:
        dN_global (torch.Tensor): Shape function gradients in global coordinates [M, 6, 3]
    """
    r, s, t = integral_point.to(device=device, dtype=dtype)
    coords = coords.to(device=device, dtype=dtype)
    elements = elements.to(device)
    jacobian = compute_c3d6_Jacobian(coords, elements, integral_point, device=device, dtype=dtype)  # [M, 3, 3]
    jacobian_inv = torch.inverse(jacobian)  # [M, 3, 3]
    
    dN_dnatural = torch.tensor([
        [ -0.5*(1 - t), -0.5*(1 - t), -0.5*(1 - r - s) ],
        [  0.5*(1 - t),  0.0,        -0.5*r         ],
        [  0.0,         0.5*(1 - t), -0.5*s         ],
        [ -0.5*(1 + t), -0.5*(1 + t),  0.5*(1 - r - s) ],
        [  0.5*(1 + t),  0.0,         0.5*r         ],
        [  0.0,         0.5*(1 + t),  0.5*s         ]
    ], dtype=dtype, device=device)  # [6, 3]
    
    dN_global = torch.einsum('mij,nj->mni', jacobian_inv, dN_dnatural)  # [M, 6, 3]
    return dN_global

def compute_c3d6_B_matrix(coords, elements, integral_point, device="cuda:0", dtype=torch.float32):
    """
    Computes the B-matrix for the C3D6 element at the given integration point.

    Args:
        coords (torch.Tensor): Node coordinates [N, 3]
        elements (torch.Tensor): Element connectivity [M, 6]
        integral_point (torch.Tensor): Natural coordinates [xi, eta, zeta]

    Returns:
        B_matrices (torch.Tensor): B matrices for all elements [M, 6, 18]
    """
    dN_global = compute_c3d6_shape_gradients(coords, elements, integral_point, device=device, dtype=dtype)  # [M, 6, 3]
    M = elements.shape[0]
    B_matrices = torch.zeros((M, 6, 18), device=device, dtype=dtype)  # [M, 6, 18]
    
    for i in range(6):
        B_matrices[:, 0, 3 * i + 0] = dN_global[:, i, 0]  # ∂N/∂x
        B_matrices[:, 1, 3 * i + 1] = dN_global[:, i, 1]  # ∂N/∂y
        B_matrices[:, 2, 3 * i + 2] = dN_global[:, i, 2]  # ∂N/∂z
        B_matrices[:, 3, 3 * i + 0] = dN_global[:, i, 1]  # ∂N/∂y
        B_matrices[:, 3, 3 * i + 1] = dN_global[:, i, 0]  # ∂N/∂x
        B_matrices[:, 4, 3 * i + 1] = dN_global[:, i, 2]  # ∂N/∂z
        B_matrices[:, 4, 3 * i + 2] = dN_global[:, i, 1]  # ∂N/∂y
        B_matrices[:, 5, 3 * i + 0] = dN_global[:, i, 2]  # ∂N/∂z
        B_matrices[:, 5, 3 * i + 2] = dN_global[:, i, 0]  # ∂N/∂x

    return B_matrices

def compute_c3d6_element_stress(coords, elements, displacement, E, nu, integral_point=None, single=True, device="cuda:0", dtype=torch.float32):
    """
    Compute element-wise stresses and von Mises stresses for C3D6 elements.

    Args:
        coords (torch.Tensor): Node coordinates [N, 3]
        elements (torch.Tensor): Element connectivity [M, 6]
        displacement (torch.Tensor): Node displacements [N, 3]
        E (float): Young's modulus
        nu (float): Poisson's ratio
        integral_point (torch.Tensor): Natural coordinates [xi, eta, zeta], optional
        single (bool): If True, returns stress at single integration point
        device (str): Device to perform computation on
        dtype (torch.dtype): Data type for computations

    Returns:
        tuple:
            stress_per_ip (torch.Tensor): Element-wise stress tensors [M, n_int, 3, 3]
            vm_stress_per_ip (torch.Tensor): Element-wise von Mises stress [M, n_int]
    """
    M = elements.shape[0]
    coords = coords.to(device=device, dtype=dtype)
    elements = elements.to(device)
    displacement = displacement.to(device=device, dtype=dtype)

    if integral_point is None:
        p, w = c3d6_integration_points(device=device, dtype=dtype)
    else:
        p, w = integral_point[:, :3].to(device=device, dtype=dtype), integral_point[:, -1].to(device=device, dtype=dtype)

    n_int = p.shape[0]

    D = compute_elasticity_matrix(E, nu, device=device, dtype=dtype)  # [6,6]

    disp_elem = displacement[elements].reshape(M, -1)  # [M, 18]

    stress_result = torch.zeros((M, n_int, 3, 3), device=device, dtype=dtype)  # [M, n_int, 3, 3]
    vm_stress_result = torch.zeros((M, n_int), device=device, dtype=dtype)    # [M, n_int]

    for idx in range(n_int):
        ip = p[idx]

        B = compute_c3d6_B_matrix(coords, elements, ip, device=device, dtype=dtype)  # [M, 6, 18]

        strain = torch.bmm(B, disp_elem.unsqueeze(2)).squeeze(2)  # [M, 6]

        stress = torch.matmul(strain, D.t())  # [M, 6]

        stress_tensor = compute_stress_tensor(stress)  # [M, 3, 3]

        element_vm_stress = compute_von_mises_stress(stress_tensor)  # [M]

        stress_result[:, idx, :, :] = stress_tensor  # [M, n_int, 3, 3]
        vm_stress_result[:, idx] = element_vm_stress  # [M, n_int]

    if single:
        stress_result = torch.einsum("i,mijk->mjk", w, stress_result)  # [M, 3, 3]
        vm_stress_result = torch.einsum("i,mi->m", w, vm_stress_result)  # [M]

    return stress_result, vm_stress_result

def compute_c3d6_K_matrix(coords, elements, E, nu, integral_point=None, single=True, device="cuda:0", dtype=torch.float32):
    """
    Computes the element stiffness matrix for each C3D6 element.

    Args:
        coords (torch.Tensor): Node coordinates [N, 3].
        elements (torch.Tensor): Element connectivity [M, 6].
        E (float): Young's modulus.
        nu (float): Poisson's ratio.
        integral_point (torch.Tensor, optional): Integration points and weights.
        single (bool, optional): If True, uses a single integration point at the centroid.
        device (str, optional): Device to perform computation on.
        dtype (torch.dtype, optional): Data type for computations.

    Returns:
        K_elements (torch.Tensor): Stiffness matrices for all elements [M, 18, 18].
    """
    M = elements.shape[0]
    coords = coords.to(device=device, dtype=dtype)
    elements = elements.to(device).type(torch.long)
    K_elements = torch.zeros((M, 18, 18), device=device, dtype=dtype)

    D = compute_elasticity_matrix(E, nu, device=device, dtype=dtype)  # [6, 6]

    if single:
        integral_point = torch.tensor([1/3, 1/3, 0.0], dtype=dtype, device=device)
        B = compute_c3d6_B_matrix(coords, elements, integral_point, device=device, dtype=dtype)  # [M, 6, 18]
        volumes = compute_wedge_volumes(coords, elements, device=device, dtype=dtype).view(M, 1, 1)  # [M, 1, 1]
        K_elements = torch.einsum('mji,jk,mkq->miq', B, D, B) * volumes
    else:
        if integral_point is None:
            integration_points, weights = c3d6_integration_points(device=device, dtype=dtype)
        else:
            integration_points, weights = integral_point[:, :3].to(device=device, dtype=dtype), integral_point[:, -1].to(device=device, dtype=dtype)

        for ip_idx in range(integration_points.shape[0]):
            ip = integration_points[ip_idx]
            B = compute_c3d6_B_matrix(coords, elements, ip, device=device, dtype=dtype)  # [M, 6, 18]
            J = compute_c3d6_Jacobian(coords, elements, ip, device=device, dtype=dtype)  # [M, 3, 3]
            detJ = torch.det(J).view(M, 1, 1)  # [M, 1, 1]
            weight = weights[ip_idx]

            K_ip = torch.einsum('mji,jk,mkq->miq', B, D, B) * detJ * weight
            K_elements += K_ip

    return K_elements  # [M, 18, 18]


##### c3d15


#################################################################################################
#########################################   쓰레기통   ############################################
#################################################################################################

##### element -> edge index 
def element_to_edge(elements, device="cuda:0"):
    """
    Generate the node adjacency matrix from tetrahedral elements.

    Args:
        elements (torch.Tensor): Tetrahedral element connectivity [M, 4]

    Returns:
        edge (torch.Tensor): Node adjacency matrix [2, E]
    """
    elements = elements.to(device)

    edge_combinations = torch.tensor([
        [0, 1], [0, 2], [0, 3],  
        [1, 2], [1, 3],        
        [2, 3]       
    ], device=device) 

    edges = elements[:, edge_combinations] 

    edges = edges.view(-1, 2) 

    edges = torch.sort(edges, dim=1)[0] 

    unique_edges = torch.unique(edges, dim=0) 

    return unique_edges.t()

##### 시각화
def visualize_surface_with_red_nodes(coords, surface_faces):
    coords_np = coords.cpu().numpy() if isinstance(coords, torch.Tensor) else np.array(coords)
    surface_faces_np = surface_faces.cpu().numpy() if isinstance(surface_faces, torch.Tensor) else np.array(surface_faces)
    
    unique_surface_nodes = np.unique(surface_faces_np)
    
    mesh = go.Mesh3d(
        x=coords_np[:, 0],
        y=coords_np[:, 1],
        z=coords_np[:, 2],
        i=surface_faces_np[:, 0],
        j=surface_faces_np[:, 1],
        k=surface_faces_np[:, 2],
        color='blue',
        opacity=0,
        name='Mesh Surface'
    )
    
    scatter = go.Scatter3d(
        x=coords_np[unique_surface_nodes, 0],
        y=coords_np[unique_surface_nodes, 1],
        z=coords_np[unique_surface_nodes, 2],
        mode='markers',
        marker=dict(color='red', size=0.5),
        name='Surface Nodes'
    )
    
    fig = go.Figure(data=[mesh, scatter])
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title='Mesh Surface with Surface Nodes Colored Red'
    )
    fig.show()

def visualize_surface_with_normals(coords, surface_faces, surface_normals, normal_scale=0.1):
    """
    Visualizes the mesh surface with surface nodes colored red and their normal vectors.

    Inputs:
        coords (torch.Tensor or numpy.ndarray): Node coordinates [N, 3]
        surface_faces (torch.Tensor or numpy.ndarray): Surface face indices [K, 3]
        surface_normals (torch.Tensor or numpy.ndarray): Normal vectors for each surface face [K, 3]
        normal_scale (float): Scaling factor for the normal vectors for visualization purposes

    Outputs:
        Displays an interactive Plotly 3D plot
    """
    coords_np = coords.cpu().numpy() if isinstance(coords, torch.Tensor) else np.array(coords)
    surface_faces_np = surface_faces.cpu().numpy() if isinstance(surface_faces, torch.Tensor) else np.array(surface_faces)
    surface_normals_np = surface_normals.cpu().numpy() if isinstance(surface_normals, torch.Tensor) else np.array(surface_normals)
    
    unique_surface_nodes = np.unique(surface_faces_np)
    
    mesh = go.Mesh3d(
        x=coords_np[:, 0],
        y=coords_np[:, 1],
        z=coords_np[:, 2],
        i=surface_faces_np[:, 0],
        j=surface_faces_np[:, 1],
        k=surface_faces_np[:, 2],
        color='lightgrey',
        opacity=0.5,
        name='Mesh Surface',
        showscale=False
    )
    
    scatter = go.Scatter3d(
        x=coords_np[unique_surface_nodes, 0],
        y=coords_np[unique_surface_nodes, 1],
        z=coords_np[unique_surface_nodes, 2],
        mode='markers',
        marker=dict(color='red', size=0.5),
        name='Surface Nodes'
    )
    
    centroids = coords_np[surface_faces_np].mean(axis=1)  # [K, 3]
    
    norms = np.linalg.norm(surface_normals_np, axis=1, keepdims=True)
    norms[norms == 0] = 1 
    normals_normalized = surface_normals_np / norms
    
    normals_scaled = normals_normalized * normal_scale
    
    cones = go.Cone(
        x=centroids[:, 0],
        y=centroids[:, 1],
        z=centroids[:, 2],
        u=normals_scaled[:, 0],
        v=normals_scaled[:, 1],
        w=normals_scaled[:, 2],
        colorscale='Reds',
        sizemode='absolute',
        sizeref=0.1,
        showscale=False,
        name='Normal Vectors'
    )
    
    fig = go.Figure(data=[mesh, scatter, cones])
    
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        title='Mesh Surface with Surface Nodes and Normal Vectors',
        legend=dict(
            itemsizing='constant'
        )
    )
    
    fig.show()

def visualize_shared_face_with_forces(coords, elements, shared_face_indices, face_id, forces, force_scale=0.1):
    """
    Visualizes two elements sharing a specified face, highlighting the shared face in red,
    other nodes and edges in green and blue respectively, and plotting the forces acting on the shared face.

    Inputs:
        coords (torch.Tensor or numpy.ndarray): Node coordinates [N, 3]
        elements (torch.Tensor or numpy.ndarray): Element connectivity [M, 4]
        shared_face_indices (torch.Tensor or numpy.ndarray): Shared face indices [S, 2, 2]
            Each shared face has two entries: [[element_id1, face_index1], [element_id2, face_index2]]
        face_id (int): Index of the shared face to visualize (0 <= face_id < S)
        forces (torch.Tensor or numpy.ndarray): Forces on each element's faces [M, 4, 3]
        force_scale (float): Scaling factor for force vectors for visualization

    Outputs:
        Displays an interactive Plotly 3D plot showing the two elements, the shared face,
        and the forces acting on the shared face.
    """
    coords_np = coords.cpu().numpy() if isinstance(coords, torch.Tensor) else np.array(coords)
    elements_np = elements.cpu().numpy() if isinstance(elements, torch.Tensor) else np.array(elements)
    shared_face_indices_np = shared_face_indices.cpu().numpy() if isinstance(shared_face_indices, torch.Tensor) else np.array(shared_face_indices)
    forces_np = forces.cpu().numpy() if isinstance(forces, torch.Tensor) else np.array(forces)

    S = shared_face_indices_np.shape[0]
    if face_id < 0 or face_id >= S:
        raise ValueError(f"face_id {face_id} is out of range [0, {S-1}]")

    shared_face = shared_face_indices_np[face_id]  # [[e1, f1], [e2, f2]]
    [e1, f1], [e2, f2] = shared_face

    face_node_indices = np.array([
        [0, 1, 2],  # Face 0
        [0, 1, 3],  # Face 1
        [1, 2, 3],  # Face 2
        [0, 2, 3]   # Face 3
    ])  # [4, 3]

    face1_nodes = elements_np[e1, face_node_indices[f1]]  # [3]
    face2_nodes = elements_np[e2, face_node_indices[f2]]  # [3]

    shared_face_nodes = np.intersect1d(face1_nodes, face2_nodes)
    if shared_face_nodes.size != 3:
        raise ValueError("Shared faces do not have the same nodes.")

    element1_nodes = elements_np[e1]  # [4]
    element2_nodes = elements_np[e2]  # [4]

    unique_nodes = np.unique(np.concatenate([element1_nodes, element2_nodes]))
    shared_nodes = shared_face_nodes
    exclusive_elem1_nodes = np.setdiff1d(element1_nodes, shared_nodes)
    exclusive_elem2_nodes = np.setdiff1d(element2_nodes, shared_nodes)

    tetra_edges = np.array([
        [0,1],
        [0,2],
        [0,3],
        [1,2],
        [1,3],
        [2,3]
    ])  # [6, 2]

    def get_element_edges(element_nodes):
        return element_nodes[tetra_edges]  # [6, 2]

    edges_elem1 = get_element_edges(element1_nodes)  # [6, 2]
    edges_elem2 = get_element_edges(element2_nodes)  # [6, 2]

    edges_elem1_sorted = np.sort(edges_elem1, axis=1)
    edges_elem2_sorted = np.sort(edges_elem2, axis=1)

    shared_face_edges = np.sort(np.array([
        [shared_nodes[0], shared_nodes[1]],
        [shared_nodes[0], shared_nodes[2]],
        [shared_nodes[1], shared_nodes[2]]
    ]), axis=1)  # [3, 2]

    def identify_shared_edges(edges, shared_face_edges):
        shared = []
        other = []
        for edge in edges:
            if np.any(np.all(edge == shared_face_edges, axis=1)):
                shared.append(edge)
            else:
                other.append(edge)
        return np.array(shared), np.array(other)

    shared_edges_elem1, other_edges_elem1 = identify_shared_edges(edges_elem1_sorted, shared_face_edges)
    shared_edges_elem2, other_edges_elem2 = identify_shared_edges(edges_elem2_sorted, shared_face_edges)

    shared_edges = np.vstack([shared_edges_elem1, shared_edges_elem2])
    other_edges = np.vstack([other_edges_elem1, other_edges_elem2])

    def unique_edges(edges):
        if edges.size == 0:
            return np.array([])
        dtype = [('n1', int), ('n2', int)]
        structured = edges.view(dtype)
        unique_structured = np.unique(structured)
        return np.array([list(edge) for edge in unique_structured])

    shared_edges_unique = unique_edges(shared_edges)
    other_edges_unique = unique_edges(other_edges)

    node_colors = []
    for node in unique_nodes:
        if node in shared_nodes:
            node_colors.append('red')
        elif node in exclusive_elem1_nodes:
            node_colors.append('green')
        elif node in exclusive_elem2_nodes:
            node_colors.append('blue')
        else:
            node_colors.append('black') 

    scatter_nodes = go.Scatter3d(
        x=coords_np[unique_nodes, 0],
        y=coords_np[unique_nodes, 1],
        z=coords_np[unique_nodes, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=node_colors,
            symbol='circle'
        ),
        name='Nodes'
    )

    def create_edge_trace(edges, color, name):
        if edges.size == 0:
            return go.Scatter3d()
        edge_x = []
        edge_y = []
        edge_z = []
        for edge in edges:
            edge_x += [coords_np[edge[0], 0], coords_np[edge[1], 0], None]
            edge_y += [coords_np[edge[0], 1], coords_np[edge[1], 1], None]
            edge_z += [coords_np[edge[0], 2], coords_np[edge[1], 2], None]
        return go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode='lines',
            line=dict(color=color, width=2),
            name=name
        )

    trace_shared_edges = create_edge_trace(shared_edges_unique, 'red', 'Shared Face Edges')
    trace_other_edges = create_edge_trace(other_edges_unique, 'green', 'Element 1 Edges') 

    shared_face_coords = coords_np[shared_nodes]
    trace_shared_face = go.Mesh3d(
        x=shared_face_coords[:, 0],
        y=shared_face_coords[:, 1],
        z=shared_face_coords[:, 2],
        i=[0],
        j=[1],
        k=[2],
        color='red',
        opacity=0.5,
        name='Shared Face',
        showscale=False
    )

    centroid_coords = shared_face_coords.mean(axis=0)  # [3]

    force1 = forces_np[e1, f1]  # [3]
    force2 = forces_np[e2, f2]  # [3]

    def normalize_vector(v):
        norm = np.linalg.norm(v)
        return v / norm if norm != 0 else v

    force1_normalized = normalize_vector(force1)
    force2_normalized = normalize_vector(force2)

    force1_scaled = force1_normalized * force_scale
    force2_scaled = force2_normalized * force_scale

    cone1 = go.Cone(
        x=[centroid_coords[0]],
        y=[centroid_coords[1]],
        z=[centroid_coords[2]],
        u=[force1_scaled[0]],
        v=[force1_scaled[1]],
        w=[force1_scaled[2]],
        colorscale='Blues',
        sizemode='absolute',
        sizeref=0.1,
        anchor='tail',
        name='Force Element 1',
        showscale=False
    )

    cone2 = go.Cone(
        x=[centroid_coords[0]],
        y=[centroid_coords[1]],
        z=[centroid_coords[2]],
        u=[force2_scaled[0]],
        v=[force2_scaled[1]],
        w=[force2_scaled[2]],
        colorscale='Reds',
        sizemode='absolute',
        sizeref=0.1,
        anchor='tail',
        name='Force Element 2',
        showscale=False
    )

    fig = go.Figure(data=[
        scatter_nodes,
        trace_shared_edges,
        trace_other_edges,
        trace_shared_face,
        cone1,
        cone2
    ])

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        title=f'Shared Face {face_id} between Element {e1} and Element {e2}',
        legend=dict(
            itemsizing='constant'
        )
    )

    fig.show()

def visualize_shared_face_with_forces_and_norm(coords, elements, shared_face_indices, face_id, forces, normals, force_scale=0.1):
    """
    Visualizes two elements sharing a specified face and the forces acting on that shared face.

    Inputs:
        coords (torch.Tensor or numpy.ndarray): Node coordinates [N, 3]
        elements (torch.Tensor or numpy.ndarray): Element connectivity [M, 4]
        shared_face_indices (torch.Tensor or numpy.ndarray): Shared face indices [S, 2, 2]
            Each shared face has two entries: [[element_id1, face_index1], [element_id2, face_index2]]
        face_id (int): Index of the shared face to visualize (0 <= face_id < S)
        forces (torch.Tensor or numpy.ndarray): Forces on each element's faces [M, 4, 3]
        normals (torch.Tensor or numpy.ndarray): Normal vectors for each element's faces [M, 4, 3]
        force_scale (float): Scaling factor for force vectors for visualization

    Outputs:
        Displays an interactive Plotly 3D plot showing the two elements, the shared face,
        and the forces acting on the shared face along with their normal vectors.
    """
    coords_np = coords.cpu().numpy() if isinstance(coords, torch.Tensor) else np.array(coords)
    elements_np = elements.cpu().numpy() if isinstance(elements, torch.Tensor) else np.array(elements)
    shared_face_indices_np = shared_face_indices.cpu().numpy() if isinstance(shared_face_indices, torch.Tensor) else np.array(shared_face_indices)
    forces_np = forces.cpu().numpy() if isinstance(forces, torch.Tensor) else np.array(forces)
    normals_np = normals.cpu().numpy() if isinstance(normals, torch.Tensor) else np.array(normals)

    S = shared_face_indices_np.shape[0]
    if face_id < 0 or face_id >= S:
        raise ValueError(f"face_id {face_id} is out of range [0, {S-1}]")

    shared_face = shared_face_indices_np[face_id]  # [[e1, f1], [e2, f2]]
    [e1, f1], [e2, f2] = shared_face

    face_node_indices = np.array([
        [0, 1, 2],  # Face 0
        [0, 1, 3],  # Face 1
        [1, 2, 3],  # Face 2
        [0, 2, 3]   # Face 3
    ])  # [4, 3]

    face1_nodes = elements_np[e1, face_node_indices[f1]]  # [3]
    face2_nodes = elements_np[e2, face_node_indices[f2]]  # [3]

    shared_face_nodes = np.intersect1d(face1_nodes, face2_nodes)
    if shared_face_nodes.size != 3:
        raise ValueError("Shared faces do not have the same nodes.")

    element1_nodes = elements_np[e1]  # [4]
    element2_nodes = elements_np[e2]  # [4]

    unique_nodes = np.unique(np.concatenate([element1_nodes, element2_nodes]))
    shared_nodes = shared_face_nodes
    exclusive_elem1_nodes = np.setdiff1d(element1_nodes, shared_nodes)
    exclusive_elem2_nodes = np.setdiff1d(element2_nodes, shared_nodes)

    tetra_edges = np.array([
        [0,1],
        [0,2],
        [0,3],
        [1,2],
        [1,3],
        [2,3]
    ])  # [6, 2]

    def get_element_edges(element_nodes):
        return element_nodes[tetra_edges]  # [6, 2]

    edges_elem1 = get_element_edges(element1_nodes)  # [6, 2]
    edges_elem2 = get_element_edges(element2_nodes)  # [6, 2]

    edges_elem1_sorted = np.sort(edges_elem1, axis=1)
    edges_elem2_sorted = np.sort(edges_elem2, axis=1)

    shared_face_edges = np.sort(np.array([
        [shared_nodes[0], shared_nodes[1]],
        [shared_nodes[0], shared_nodes[2]],
        [shared_nodes[1], shared_nodes[2]]
    ]), axis=1)  # [3, 2]

    def identify_shared_edges(edges, shared_face_edges):
        shared = []
        other = []
        for edge in edges:
            if np.any(np.all(edge == shared_face_edges, axis=1)):
                shared.append(edge)
            else:
                other.append(edge)
        return np.array(shared), np.array(other)

    shared_edges_elem1, other_edges_elem1 = identify_shared_edges(edges_elem1_sorted, shared_face_edges)
    shared_edges_elem2, other_edges_elem2 = identify_shared_edges(edges_elem2_sorted, shared_face_edges)

    shared_edges = np.vstack([shared_edges_elem1, shared_edges_elem2])
    other_edges = np.vstack([other_edges_elem1, other_edges_elem2])

    def unique_edges(edges):
        if edges.size == 0:
            return np.array([])
        dtype = [('n1', int), ('n2', int)]
        structured = edges.view(dtype)
        unique_structured = np.unique(structured)
        return np.array([list(edge) for edge in unique_structured])

    shared_edges_unique = unique_edges(shared_edges)
    other_edges_unique = unique_edges(other_edges)

    node_colors = []
    for node in unique_nodes:
        if node in shared_nodes:
            node_colors.append('red')
        elif node in exclusive_elem1_nodes:
            node_colors.append('green')
        elif node in exclusive_elem2_nodes:
            node_colors.append('blue')
        else:
            node_colors.append('black')  

    scatter_nodes = go.Scatter3d(
        x=coords_np[unique_nodes, 0],
        y=coords_np[unique_nodes, 1],
        z=coords_np[unique_nodes, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=node_colors,
            symbol='circle'
        ),
        name='Nodes'
    )

    # Prepare edge traces
    def create_edge_trace(edges, color, name):
        if edges.size == 0:
            return go.Scatter3d()
        edge_x = []
        edge_y = []
        edge_z = []
        for edge in edges:
            edge_x += [coords_np[edge[0], 0], coords_np[edge[1], 0], None]
            edge_y += [coords_np[edge[0], 1], coords_np[edge[1], 1], None]
            edge_z += [coords_np[edge[0], 2], coords_np[edge[1], 2], None]
        return go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode='lines',
            line=dict(color=color, width=2),
            name=name
        )

    trace_shared_edges = create_edge_trace(shared_edges_unique, 'red', 'Shared Face Edges')
    trace_other_edges = create_edge_trace(other_edges_unique, 'green', 'Other Edges')

    shared_face_coords = coords_np[shared_nodes]
    trace_shared_face = go.Mesh3d(
        x=shared_face_coords[:, 0],
        y=shared_face_coords[:, 1],
        z=shared_face_coords[:, 2],
        i=[0],
        j=[1],
        k=[2],
        color='red',
        opacity=0.5,
        name='Shared Face',
        showscale=False
    )

    centroid_coords = shared_face_coords.mean(axis=0)  # [3]

    force1 = forces_np[e1, f1]  # [3]
    force2 = forces_np[e2, f2]  # [3]

    def normalize_vector(v):
        norm = np.linalg.norm(v)
        return v / norm if norm != 0 else v

    force1_normalized = normalize_vector(force1)
    force2_normalized = normalize_vector(force2)

    force1_scaled = force1_normalized * force_scale
    force2_scaled = force2_normalized * force_scale

    cone1 = go.Cone(
        x=[centroid_coords[0]],
        y=[centroid_coords[1]],
        z=[centroid_coords[2]],
        u=[force1_scaled[0]],
        v=[force1_scaled[1]],
        w=[force1_scaled[2]],
        colorscale='Blues',
        sizemode='absolute',
        sizeref=0.1,
        anchor='tail',
        name='Force Element 1',
        showscale=False
    )

    cone2 = go.Cone(
        x=[centroid_coords[0]],
        y=[centroid_coords[1]],
        z=[centroid_coords[2]],
        u=[force2_scaled[0]],
        v=[force2_scaled[1]],
        w=[force2_scaled[2]],
        colorscale='Reds',
        sizemode='absolute',
        sizeref=0.1,
        anchor='tail',
        name='Force Element 2',
        showscale=False
    )

    normal1 = normals_np[e1, f1]  # [3]
    normal2 = normals_np[e2, f2]  # [3]

    normal1_normalized = normalize_vector(normal1)
    normal2_normalized = normalize_vector(normal2)

    normal_scale = force_scale
    normal1_scaled = normal1_normalized * normal_scale
    normal2_scaled = normal2_normalized * normal_scale

    cone3 = go.Cone(
        x=[centroid_coords[0]],
        y=[centroid_coords[1]],
        z=[centroid_coords[2]],
        u=[normal1_scaled[0]],
        v=[normal1_scaled[1]],
        w=[normal1_scaled[2]],
        colorscale='Greens',
        sizemode='absolute',
        sizeref=0.1,
        anchor='tail',
        name='Normal Element 1',
        showscale=False
    )

    cone4 = go.Cone(
        x=[centroid_coords[0]],
        y=[centroid_coords[1]],
        z=[centroid_coords[2]],
        u=[normal2_scaled[0]],
        v=[normal2_scaled[1]],
        w=[normal2_scaled[2]],
        colorscale='Greens',
        sizemode='absolute',
        sizeref=0.1,
        anchor='tail',
        name='Normal Element 2',
        showscale=False
    )

    fig = go.Figure(data=[
        scatter_nodes,
        trace_shared_edges,
        trace_other_edges,
        trace_shared_face,
        cone1,
        cone2,
        cone3,
        cone4
    ])

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        title=f'Shared Face {face_id} between Element {e1} and Element {e2}',
        legend=dict(
            itemsizing='constant'
        )
    )

    fig.show()

##### 계산
def compute_c3d4_surface_forces(normal_vectors, stress_tensors, device="cuda:0"):
    """
    element normal vector(면적포함), element stress tensor를 받아 element face별 알짜힘을 구하는 함수
    이때 element normal vector tensor는 기존 element tensor와 사면체 순서가 동일하도록 설정해주어야한다. 

    Inputs:
        normal_vectors (torch.Tensor): Normal vectors for each face of each element [M, 4, 3]
        stress_tensors (torch.Tensor): Stress tensors for each element [M, 3, 3]

    Outputs:
        face forces (torch.Tensor): Forces on each face of each element [M, 4, 3]
    """
    stress_tensors_expanded = stress_tensors.unsqueeze(1).to(device)  # [M, 1, 3, 3]
    normal_vectors= normal_vectors.unsqueeze(-1).to(device) # [M, 4, 3, 1]

    face_forces = torch.matmul(stress_tensors_expanded, normal_vectors).squeeze(-1)  # [M, 4, 3]

    return face_forces

def compute_c3d4_shared_face_forces_sum(shared_face_indices, element_forces, device="cuda:0"):
    """
    각 element face forces 정보, 그리고 element간 face 공유 정보가 주어지면 힘평형에 대한 조건을 구하는 함수
    모든 값이 0이여야 힘 평형에 도달한 것이다.

    Inputs:
        shared_face_indices (torch.Tensor): Indices of elements sharing each face [S, 2, 2]
        element_forces (torch.Tensor): Forces on each face of each element [M, 4, 3]

    Outputs:
        shared_face_forces_sum (torch.Tensor): Sum of forces on each shared face [S, 3]
    """
    shared_face_indices = shared_face_indices.to(device)
    element_forces = element_forces.to(device)

    force_1 = element_forces[shared_face_indices[:, 0, 0], shared_face_indices[:, 0, 1], :]  # [S, 3]
    force_2 = element_forces[shared_face_indices[:, 1, 0], shared_face_indices[:, 1, 1], :]  # [S, 3]

    shared_face_forces_sum = force_1 + force_2  # [S, 3]

    return shared_face_forces_sum

