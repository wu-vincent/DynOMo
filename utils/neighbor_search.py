import faiss
import faiss.contrib.torch_utils
import torch


def torch_3d_knn(pts, num_knn, method="l2"):
    # Initialize FAISS index
    if method == "l2":
        index = faiss.IndexFlatL2(pts.shape[1])
    elif method == "cosine":
        index = faiss.IndexFlatIP(pts.shape[1])
    else:
        raise NotImplementedError(f"Method: {method}")

    # Convert FAISS index to GPU
    if pts.get_device() != -1:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    # Add points to index and compute distances
    index.add(pts)
    distances, indices = index.search(pts, num_knn)
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


def calculate_neighbors_seg(params, variables, time_idx, instseg_mask, num_knn=20):
    device = params['means3D'].device
    indices = torch.zeros(params['means3D'].shape[0], num_knn).long().to(device)
    weight = torch.zeros(params['means3D'].shape[0], num_knn).to(device)
    dist = torch.zeros(params['means3D'].shape[0], num_knn).to(device)
    for inst in instseg_mask.unique():
        bin_mask = instseg_mask == inst
        if time_idx is None:
            pts = params['means3D'].detach()
        else:
            pts = params['means3D'][:, :, time_idx].detach()
        pts = pts[bin_mask]
        neighbor_dist, neighbor_indices = torch_3d_knn(pts.contiguous(), num_knn+1)
        neighbor_dist = neighbor_dist[:, 1:]
        neighbor_indices = neighbor_indices[:, 1:]
        neighbor_weight = torch.exp(-2000 * torch.square(neighbor_dist))
        indices[bin_mask] = neighbor_indices
        weight[bin_mask] = neighbor_weight
        dist[bin_mask] = neighbor_dist
    variables["self_indices"] = torch.arange(params['means3D'].shape[0]).unsqueeze(1).tile(num_knn).flatten()
    variables["neighbor_indices"] = indices.flatten().long().contiguous()
    variables["neighbor_weight"] = weight.flatten().float().contiguous()
    variables["neighbor_dist"] = dist.flatten().float().contiguous()
    return variables