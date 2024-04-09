import faiss
import faiss.contrib.torch_utils
import torch


def torch_3d_knn(q_pts, k_pts=None, num_knn=20, method="l2"):
    # If query and key points are the same set
    if k_pts is None:
        k_pts = q_pts

    # Initialize FAISS index
    if method == "l2":
        index = faiss.IndexFlatL2(q_pts.shape[1])
    elif method == "cosine":
        index = faiss.IndexFlatIP(q_pts.shape[1])
    else:
        raise NotImplementedError(f"Method: {method}")

    # Convert FAISS index to GPU
    if q_pts.get_device() != -1:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    # Add points to index and compute distances
    index.add(k_pts)
    distances, indices = index.search(q_pts, num_knn)
    return distances, indices
    

def calculate_neighbors(params, variables, time_idx, num_knn=20):
    if time_idx is None:
        pts = params['means3D'].detach()
    else:
        pts = params['means3D'][:, :, time_idx].detach()
    neighbor_dist, neighbor_indices = torch_3d_knn(pts.contiguous(), num_knn)
    neighbor_weight = torch.exp(-2000 * torch.square(neighbor_dist))
    variables["neighbor_indices"] = neighbor_indices.long().contiguous()
    variables["neighbor_weight"] = neighbor_weight.float().contiguous()
    variables["neighbor_dist"] = neighbor_dist.float().contiguous()
    return variables


def calculate_neighbors_seg(
        params, variables, time_idx, instseg_mask, num_knn=20, existing_params=None, \
            existing_instseg_mask=None, dist_to_use='rgb', use_old=False):
    embeddings_in_params = 'embeddings' in params.keys()
    device = params['means3D'].device
    if existing_params is not None:
        number_existing_gaussians = existing_params['means3D'].shape[0]

    # initalize matrices
    indices = torch.zeros(params['means3D'].shape[0], num_knn).long().to(device)
    weight = torch.zeros(params['means3D'].shape[0], num_knn).to(device)
    dist = torch.zeros(params['means3D'].shape[0], num_knn).to(device)
    
    # get existing Gaussians and neighbor arranged indices
    if existing_params is not None:
        if use_old:
            if embeddings_in_params:
                existing_embeddings = torch.cat(
                    [existing_params['embeddings'].detach(), params['embeddings'].detach()])
            existing_colors = torch.cat(
                [existing_params['rgb_colors'].detach(), params['rgb_colors'].detach()])
            if len(existing_params['means3D'].shape) == 3:
                existing_means = torch.cat(
                    [existing_params['means3D'][:, :, time_idx].detach().contiguous(),
                    params['means3D'][:, :, time_idx].detach()])
            else:
                existing_means = torch.cat(
                    [existing_params['means3D'][:, :].detach().contiguous(),
                    params['means3D'][:, :].detach()])
            aranged_idx = torch.arange(existing_means.shape[0]).to(device)
            existing_instseg_mask = torch.cat([existing_instseg_mask, instseg_mask])
        else:
            existing_colors = existing_params['rgb_colors'].detach()
            if embeddings_in_params:
                existing_embeddings = existing_params['embeddings'].detach()
            if len(existing_params['means3D'].shape) == 3:
                existing_means = existing_params['means3D'][:, :, time_idx].detach().contiguous()
            else:
                existing_means = existing_params['means3D'][:, :].detach().contiguous()
            aranged_idx = torch.arange(existing_means.shape[0]).to(device)
            existing_instseg_mask = existing_instseg_mask
    else:
        existing_means = None
        aranged_idx = torch.arange(params['means3D'].shape[0]).to(device)
    
    # Iterate over segment IDs
    for inst in instseg_mask.unique():
        # mask query points per segment
        bin_mask = instseg_mask == inst
        if len(params['means3D'].shape) == 3:
            q_pts = params['means3D'][:, :, time_idx].detach()
        else:
            q_pts = params['means3D'].detach()            
        q_pts = q_pts[bin_mask]
        q_colors = params['rgb_colors'][bin_mask].detach()
        if embeddings_in_params:
            q_embeddings = params['embeddings'][bin_mask].detach()

        # mask key points
        if existing_params is not None:
            k_bin_mask = existing_instseg_mask == inst
            k_pts = existing_means[k_bin_mask].contiguous()
            k_colors = existing_colors[k_bin_mask]
            if embeddings_in_params:
                k_embeddings = existing_embeddings[k_bin_mask]
        else:
            k_pts = q_pts
            k_colors = q_colors
            if embeddings_in_params:
                k_embeddings = q_embeddings

        # get distances and indices
        neighbor_dist, neighbor_indices = torch_3d_knn(q_pts.contiguous(), k_pts, num_knn=num_knn+num_knn+1)
        # calculate weight of neighbors
        if dist_to_use == 'l2':
            neighbor_dist = neighbor_dist[:, 1:num_knn+1]
        elif dist_to_use == 'rgb':
            # print(neighbor_dist.shape)
            k_colors = k_colors[neighbor_indices[:, 1:num_knn+1]]
            q_colors = q_colors.unsqueeze(1)
            neighbor_dist = torch.cdist(q_colors, k_colors).squeeze()
        elif dist_to_use == 'dinov2':
            k_embeddings = k_embeddings[neighbor_indices[:, 1:num_knn+1]]
            q_embeddings = q_embeddings.unsqueeze(1)
            neighbor_dist = torch.cdist(
                torch.nn.functional.normalize(q_embeddings, dim=2),
                torch.nn.functional.normalize(k_embeddings, dim=2)).squeeze()

        if existing_params is not None:
            num_samps = neighbor_indices.shape[0]
            neighbor_indices = aranged_idx[k_bin_mask][neighbor_indices[:, 1:num_knn+1].flatten()]
            neighbor_indices = neighbor_indices.reshape((num_samps, num_knn))
        else:
            neighbor_indices = aranged_idx[bin_mask][neighbor_indices[:, 1:num_knn+1]]

        neighbor_weight = torch.nn.functional.softmax(-neighbor_dist, dim=1)
        # neighbor_weight = torch.exp(-2000 * torch.square(neighbor_dist))
        indices[bin_mask] = neighbor_indices
        weight[bin_mask] = neighbor_weight

    if existing_params is not None:
        variables["self_indices"] = torch.arange(
            params['means3D'].shape[0]).unsqueeze(1).tile(num_knn).flatten().to(device) + number_existing_gaussians
    else:
        variables["self_indices"] = torch.arange(
            params['means3D'].shape[0]).unsqueeze(1).tile(num_knn).flatten().to(device)

    non_neighbor_mask = indices.flatten() != -1
    variables["neighbor_indices"] = indices.flatten().long().contiguous()[non_neighbor_mask]
    variables["neighbor_weight"] = weight.flatten().float().contiguous()[non_neighbor_mask]
    variables["neighbor_dist"] = dist.flatten().float().contiguous()[non_neighbor_mask]
    variables["self_indices"] = variables["self_indices"][non_neighbor_mask]
    
    return variables