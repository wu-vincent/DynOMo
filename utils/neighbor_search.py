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


def calculate_neighbors_seg_after_init(
        params,
        variables,
        time_idx,
        num_knn=20,
        dist_to_use='rgb',
        use_old_and_new=True,
        inflate=2,
        l2_thresh=0.5):
    
    if time_idx != 0:
        new_params = dict()
        old_params = dict()
        time_mask = variables['timestep'] < time_idx
        for p, v in params.items():
            if 'embeddings' in p or 'means3D' in p or 'rgb_colors' in p:
                old_params[p] = v[time_mask]
                new_params[p] = v[~time_mask]
        existing_instseg_mask = params['instseg'][time_mask]
        instseg_mask = params['instseg'][~time_mask]
    else:
        new_params = params
        instseg_mask = params['instseg']
        old_params = None
        existing_instseg_mask = None

    new_variables = dict()
    new_variables, to_remove = calculate_neighbors_seg(
            new_params,
            new_variables,
            time_idx,
            instseg_mask,
            num_knn=num_knn,
            existing_params=old_params,
            existing_instseg_mask=existing_instseg_mask,
            dist_to_use=dist_to_use,
            use_old_and_new=use_old_and_new,
            inflate=inflate,
            l2_thresh=l2_thresh)
    if time_idx != 0:
        variables['self_indices'] = torch.cat((variables['self_indices'], new_variables['self_indices']), dim=0)
        variables['neighbor_indices'] = torch.cat((variables['neighbor_indices'], new_variables['neighbor_indices']), dim=0)
        variables['neighbor_weight'] = torch.cat((variables['neighbor_weight'], new_variables['neighbor_weight']), dim=0)
        variables['neighbor_weight_sm'] = torch.cat((variables['neighbor_weight_sm'], new_variables['neighbor_weight_sm']), dim=0)
        variables['neighbor_dist'] = torch.cat((variables['neighbor_dist'], new_variables['neighbor_dist']), dim=0)
    else:
        variables.update(new_variables)

    return variables            


def calculate_neighbors_seg(
        params,
        variables,
        time_idx,
        instseg_mask,
        num_knn=20,
        existing_params=None,
        existing_instseg_mask=None,
        dist_to_use='rgb',
        use_old_and_new=True,
        inflate=2,
        l2_thresh=0.5):

    embeddings_in_params = 'embeddings' in params.keys()
    device = params['means3D'].device
    if existing_params is not None:
        number_existing_gaussians = existing_params['means3D'].shape[0]

    # initalize matrices
    indices = torch.zeros(params['means3D'].shape[0], num_knn).long().to(device)
    weight = torch.zeros(params['means3D'].shape[0], num_knn).to(device)
    weight_sm = torch.zeros(params['means3D'].shape[0], num_knn).to(device)
    dist = torch.zeros(params['means3D'].shape[0], num_knn).to(device)
    to_remove = torch.zeros(params['means3D'].shape[0], dtype=bool).to(device)

    # get existing Gaussians and neighbor arranged indices
    if existing_params is not None:
        if use_old_and_new:
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
        q_colors = torch.nn.functional.normalize(params['rgb_colors'][bin_mask], p=2, dim=1).detach()
        if embeddings_in_params:
            q_embeddings = torch.nn.functional.normalize(params['embeddings'][bin_mask], p=2, dim=1).detach()

        # mask key points
        if existing_params is not None:
            k_bin_mask = existing_instseg_mask == inst
            k_pts = existing_means[k_bin_mask].contiguous()
            k_colors = torch.nn.functional.normalize(existing_colors[k_bin_mask], p=2, dim=1)
            if embeddings_in_params:
                k_embeddings = torch.nn.functional.normalize(existing_embeddings[k_bin_mask], p=2, dim=1)
        else:
            k_pts = q_pts
            k_colors = q_colors
            if embeddings_in_params:
                k_embeddings = q_embeddings

        # get distances and indices
        neighbor_dist, neighbor_indices = torch_3d_knn(
            q_pts.contiguous(), k_pts, num_knn=int(inflate*num_knn)+1)
        to_remove_seg = neighbor_dist[:, 1:num_knn+1].min(dim=1).values > l2_thresh

        l2_neighbor_dists = neighbor_dist[:, 1:num_knn+1]

        # calculate weight of neighbors
        if dist_to_use == 'l2':
            neighbor_dist = neighbor_dist[:, 1:num_knn+1]
            neighbor_indices = neighbor_indices[:, 1:num_knn+1]
        elif dist_to_use == 'rgb' or dist_to_use == 'embeddings':
            # get rbg distance from nearest neighbors in l2
            neighbor_indices = neighbor_indices[:, 1:]
            q_idx = torch.tile(
                torch.arange(neighbor_indices.shape[0]).unsqueeze(1),
                (1, int(inflate*num_knn))).flatten()
            if dist_to_use == 'rgb':
                neighbor_dist = torch.cdist(
                        q_colors.float()[q_idx, :].unsqueeze(1),
                        k_colors.float()[neighbor_indices.flatten(), :].unsqueeze(1)
                    ).squeeze()
            else:
                neighbor_dist = torch.cdist(
                        q_embeddings.float()[q_idx, :].unsqueeze(1),
                        k_embeddings.float()[neighbor_indices.flatten(), :].unsqueeze(1)
                    ).squeeze()
            neighbor_dist = neighbor_dist.reshape(neighbor_indices.shape[0], -1)

            # sort rgb neighbot distance and re-index to get closest points in 
            # wrt rgb within closest points in l2
            neighbor_dist = neighbor_dist.sort(dim=1, descending=False)
            idx = neighbor_dist.indices[:, :num_knn]
            q_idx = torch.tile(torch.arange(neighbor_indices.shape[0]).unsqueeze(1), (1, num_knn)).flatten()
            neighbor_indices = neighbor_indices[
                q_idx, idx.flatten()].squeeze().reshape(q_colors.shape[0], num_knn)
            neighbor_dist = neighbor_dist.values[:, :num_knn]
            # quit()

        if existing_params is not None:
            num_samps = neighbor_indices.shape[0]
            neighbor_indices = aranged_idx[k_bin_mask][neighbor_indices.flatten()]
            neighbor_indices = neighbor_indices.reshape((num_samps, num_knn))
        else:
            neighbor_indices = aranged_idx[bin_mask][neighbor_indices]
        
        neighbor_weight_sm = torch.nn.functional.softmax(-neighbor_dist, dim=1)
        neighbor_weight = torch.exp(-2000 * torch.square(neighbor_dist))
        indices[bin_mask] = neighbor_indices
        weight[bin_mask] = neighbor_weight
        weight_sm[bin_mask] =neighbor_weight_sm
        to_remove[bin_mask] = to_remove_seg

    if existing_params is not None:
        variables["self_indices"] = torch.arange(
            params['means3D'].shape[0]).unsqueeze(1).tile(num_knn).flatten().to(device) + number_existing_gaussians
    else:
        variables["self_indices"] = torch.arange(
            params['means3D'].shape[0]).unsqueeze(1).tile(num_knn).flatten().to(device)

    non_neighbor_mask = indices.flatten() != -1
    variables["neighbor_indices"] = indices.flatten().long().contiguous()[non_neighbor_mask]
    variables["neighbor_weight"] = weight.flatten().float().contiguous()[non_neighbor_mask]
    variables["neighbor_weight_sm"] = weight_sm.flatten().float().contiguous()[non_neighbor_mask]
    variables["neighbor_dist"] = dist.flatten().float().contiguous()[non_neighbor_mask]
    variables["self_indices"] = variables["self_indices"][non_neighbor_mask]
    
    return variables, to_remove


def calculate_neighbors_between_pc(
        params, time_idx, other_params=None, other_time_idx=None, num_knn=20, dist_to_use='rgb', inflate=2):
    embeddings_in_params = 'embeddings' in other_params.keys()
    device = params['means3D'].device

    # initalize matrices
    indices = torch.zeros(params['means3D'].shape[0], num_knn).long().to(device)
    weight = torch.zeros(params['means3D'].shape[0], num_knn).to(device)
    dist = torch.zeros(params['means3D'].shape[0], num_knn).to(device)
    
    # get existing Gaussians and neighbor arranged indices
    if other_params is not None:
        other_colors = other_params['rgb_colors'].detach()
        if embeddings_in_params:
            other_embeddings = other_params['embeddings'].detach()
        if len(other_params['means3D'].shape) == 3:
            other_means = other_params['means3D'][:, :, other_time_idx].detach().contiguous()
        else:
            other_means = other_params['means3D'].detach().contiguous()
        aranged_idx = torch.arange(other_means.shape[0]).to(device)
    else:
        other_means = None
        aranged_idx = torch.arange(params['means3D'].shape[0]).to(device)

    # Iterate over segment IDs
    # mask query points per segment
    if len(params['means3D'].shape) == 3:
        q_pts = params['means3D'][:, :, time_idx].detach()
    else:
        q_pts = params['means3D'].detach()            
    q_colors = torch.nn.functional.normalize(params['rgb_colors'], p=2, dim=1).detach()
    if embeddings_in_params:
        q_embeddings = torch.nn.functional.normalize(params['embeddings'], p=2, dim=1).detach()

    # mask key points
    if other_params is not None:
        k_pts = other_means.contiguous()
        k_colors = torch.nn.functional.normalize(other_colors, p=2, dim=1)
        if embeddings_in_params:
            k_embeddings = torch.nn.functional.normalize(other_embeddings, p=2, dim=1)
    else:
        k_pts = q_pts
        k_colors = q_colors
        if embeddings_in_params:
            k_embeddings = q_embeddings

    # get distances and indices
    neighbor_dist, neighbor_indices = torch_3d_knn(
        q_pts.contiguous(), k_pts, num_knn=int(inflate*num_knn))
    # calculate weight of neighbors
    if dist_to_use == 'l2':
        neighbor_dist = neighbor_dist[:, :num_knn]
        neighbor_indices = neighbor_indices[:, :num_knn]
    elif dist_to_use == 'rgb' or dist_to_use == 'embeddings':
        # get rbg distance from nearest neighbors in l2
        q_idx = torch.tile(
            torch.arange(neighbor_indices.shape[0]).unsqueeze(1),
            (1, int(inflate*num_knn))).flatten()
        if dist_to_use == 'rgb':
            neighbor_dist = torch.cdist(
                    q_colors.float()[q_idx, :].unsqueeze(1),
                    k_colors.float()[neighbor_indices.flatten(), :].unsqueeze(1)
                ).squeeze()
        else:
            neighbor_dist = torch.cdist(
                    q_embeddings.float()[q_idx, :].unsqueeze(1),
                    k_embeddings.float()[neighbor_indices.flatten(), :].unsqueeze(1)
                ).squeeze()
        neighbor_dist = neighbor_dist.reshape(neighbor_indices.shape[0], -1)
        # sort rgb neighbot distance and re-index to get closest points in 
        # wrt rgb within closest points in l2
        neighbor_dist = neighbor_dist.sort(dim=1, descending=False)
        idx = neighbor_dist.indices[:, :num_knn]
        q_idx = torch.tile(torch.arange(neighbor_indices.shape[0]).unsqueeze(1), (1, num_knn)).flatten()
        neighbor_indices = neighbor_indices[
            q_idx, idx.flatten()].squeeze().reshape(q_colors.shape[0], num_knn)
        neighbor_dist = neighbor_dist.values[:, :num_knn]

    neighbor_weight_sm = torch.nn.functional.softmax(-torch.atleast_2d(neighbor_dist), dim=1).squeeze()
    neighbor_weight = torch.exp(-2000 * torch.square(neighbor_dist))
    neighbor_dict = dict()
    neighbor_dict["self_indices"] = torch.arange(
        params['means3D'].shape[0]).unsqueeze(1).tile(num_knn).flatten().to(device)
    neighbor_dict["neighbor_indices"] = neighbor_indices.flatten().long().contiguous()
    neighbor_dict["neighbor_weight"] = neighbor_weight.flatten().float().contiguous()
    neighbor_dict["neighbor_dist"] = neighbor_dist.flatten().float().contiguous()
    neighbor_dict["neighbor_weight_sm"] = neighbor_weight_sm.flatten().float().contiguous()

    return neighbor_dict
