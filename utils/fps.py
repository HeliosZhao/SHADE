
import torch
import numpy as np

def farthest_point_sample_tensor(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D] 
        npoint: number of samples
        dim: dimension for point, 2 for tsne, 6 for img mean concate img std
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    device = point.device
    N, D = point.shape
    xyz = point
    centroids = torch.zeros((npoint,), device=device)
    distance = torch.ones((N,), device=device) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.argmax(distance, dim=-1)
    point = point[centroids.long()]
    return point, centroids.long()


    

