from ..colorspace import rgb_to_od, od_to_hsd

from typing import Tuple, Union, Optional
from kmeans_pytorch import kmeans
import torch


class UnmixException(Exception):
    pass


def polar_sort(clusters: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    # TODO: sort by distance after to break ties
    cluster_labels, cluster_centers = clusters
    origin = cluster_centers.mean(dim=0)
    cluster_centers_adj = cluster_centers - origin
    polar_angles = torch.atan2(cluster_centers_adj[:, 1], cluster_centers_adj[:, 0])
    sorted_idx = torch.argsort(polar_angles)
    return sorted_idx[cluster_labels], cluster_centers[sorted_idx]


def cluster_hsd(hsd: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    cx, cy, od_mu = hsd
    if 'num_clusters' not in kwargs:
        kwargs['num_clusters'] = 2
    if 'device' not in kwargs:
        kwargs['device'] = cx.device
    if 'tqdm_flag' not in kwargs:
        kwargs['tqdm_flag'] = False
    x = torch.stack([cx, cy], dim=1)
    clusters = kmeans(X=x, **kwargs)
    # TODO: pytorch_kmeans forces results onto the cpu: https://github.com/subhadarship/kmeans_pytorch/issues/27
    cluster_labels, cluster_centers = clusters
    return polar_sort((cluster_labels.to(cx.device), cluster_centers.to(cx.device)))


def threshold_od(hsd: torch.Tensor, thresh: float) -> Tuple[torch.Tensor, torch.Tensor]:
    cx = hsd[:, 0, :, :]
    cy = hsd[:, 1, :, :]
    od_mu = hsd[:, 2, :, :]
    thresh_mask = od_mu > thresh
    thresh_mask_inv = ~thresh_mask
    return (torch.stack([cx[thresh_mask], cy[thresh_mask], od_mu[thresh_mask]]),
            torch.stack([cx[thresh_mask_inv], cy[thresh_mask_inv], od_mu[thresh_mask_inv]]))


def calc_distance(clusters: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    cluster_centers = clusters[1] if not torch.is_tensor(clusters) else clusters
    n_clusters = cluster_centers.shape[0]
    if n_clusters == 2:
        return torch.linalg.norm(cluster_centers[0] - cluster_centers[1])
    else:
        x1, y1 = cluster_centers[0]
        x2, y2 = cluster_centers[1]
        x3, y3 = cluster_centers[2]
        return torch.abs(0.5 * (((x2 - x1) * (y3 - y1)) - ((x3 - x1) * (y2 - y1))))


def threshold_distances(distances: torch.Tensor, percentile: float = 0.99) -> Tuple[torch.Tensor, torch.Tensor]:
    thresh = distances.quantile(torch.tensor([percentile]).to(distances.device))
    thresh_idx = torch.where(distances >= thresh)[0]
    thresh_idx_inv = torch.where(distances < thresh)[0]
    return thresh_idx, thresh_idx_inv


def unmix_batch(rgb: torch.Tensor,
                threshold: float = 0.3,
                **kwargs) -> Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]]:
    hsd = od_to_hsd(rgb_to_od(rgb))
    hsd, hsd_noise = threshold_od(hsd, threshold)
    if hsd.shape[1] == 0:
        raise UnmixException("Unable to process image with no relevant pixel data after thresholding")
    clusters = cluster_hsd(hsd, **kwargs)
    return clusters, hsd, hsd_noise


def cluster_medians(clusters: torch.Tensor) -> torch.Tensor:
    return clusters.quantile(0.5, dim=0)
