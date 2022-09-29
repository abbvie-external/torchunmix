from .functional import unmix_batch, calc_distance, threshold_distances, polar_sort, cluster_medians
from ..colorspace import hsd_to_od
from .plot import plot as plot_

from torch.utils.data import DataLoader
from typing import Tuple
from tqdm import tqdm
import torch


class Unmix(object):
    def __init__(self,
                 batches: DataLoader,
                 threshold: float = 0.3,
                 percentile: float = 0.99,
                 progress: bool = True,
                 device: str = 'cpu',
                 **kwargs):

        # public
        self.percentile = percentile

        # private
        self._batches = batches
        self._threshold = threshold
        self._progress = progress
        self._kwargs = kwargs
        self._cluster_centers = []
        self._distances = []
        self._device = device

        self._process_batches()

    def _process_batch(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        clusters, hsd, noise = unmix_batch(batch, self._threshold, **self._kwargs)
        _, cluster_centers = clusters
        return cluster_centers, calc_distance(cluster_centers)

    def _process_batches(self):
        for batch in tqdm(self._batches, total=len(self._batches), disable=not self._progress):
            cluster_centers, distance = self._process_batch(batch.to(self._device))
            self._cluster_centers.append(cluster_centers.tolist())
            self._distances.append(calc_distance(cluster_centers))
        self._cluster_centers = torch.tensor(self._cluster_centers, device=self._device)

    def _threshold_centers(self) -> Tuple[torch.Tensor, torch.Tensor]:
        include_idxs, exclude_idxs = threshold_distances(
            torch.tensor(self._distances, device=self._distances[0].device),
            self.percentile)
        include_clusters = self._cluster_centers[include_idxs]
        exclude_clusters = self._cluster_centers[exclude_idxs]
        return include_clusters, exclude_clusters

    def plot(self):
        include_clusters, exclude_clusters = self._threshold_centers()
        plot_(cluster_medians(include_clusters),
              include_clusters.reshape(-1, 2).permute(1, 0),
              exclude_clusters.reshape(-1, 2).permute(1, 0))

    def stains(self) -> Tuple[torch.Tensor, torch.Tensor]:
        stains = hsd_to_od(cluster_medians(self._threshold_centers()[0]))
        if stains.shape[0] == 2:
            stains = torch.vstack([stains, torch.zeros(3, device=stains.device)])
            stains[2, :] = torch.cross(stains[0, :], stains[1, :])
        stains = stains / torch.sqrt(torch.sum(torch.square(stains), dim=1))
        return torch.linalg.inv(stains), stains
