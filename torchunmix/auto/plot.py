from ..colorspace import hsd_to_od, od_to_rgb

import matplotlib.pyplot as plt
import torch


def plot_chromacity_triangle(**kwargs) -> None:
    plot_kwargs = {
        'c': 'black',
        'linewidth': 2,
        'linestyle': 'dashed'
    }
    plot_kwargs = {**plot_kwargs, **kwargs}
    plt.plot((-1.0, 2.0, -1.0, -1.0), (2.0, 0.0, -2.0, 2.0), **plot_kwargs)
    plt.xlim([-1.1, 2.1])
    plt.ylim([-2.1, 2.1])


def plot_cluster_centers(cluster_centers: torch.Tensor, **kwargs) -> None:
    plot_kwargs = {
        'marker': '+',
        'markersize': 7,
        'markeredgecolor': 'lime',
    }
    cluster_centers = cluster_centers.cpu().numpy()
    plot_kwargs = {**plot_kwargs, **kwargs}
    for i in range(cluster_centers.shape[0]):
        plt.plot(cluster_centers[i][0], cluster_centers[i][1], **plot_kwargs)


def plot_exclude_hsd(noise: torch.Tensor, **kwargs) -> None:
    plot_kwargs = {
        'c': 'gainsboro',
        's': 150,
        'marker': 'o'
    }
    plot_kwargs = {**plot_kwargs, **kwargs}
    cx, cy = noise
    plt.scatter(x=cx.cpu().numpy(), y=cy.cpu().numpy(), **plot_kwargs)


def plot_include_hsd(hsd: torch.Tensor, **kwargs) -> None:
    plot_kwargs = {
        's': 150,
        'marker': 'o',
        'c': (od_to_rgb(hsd_to_od(hsd.permute(1, 0))) / 255).cpu().numpy()
    }
    plot_kwargs = {**plot_kwargs, **kwargs}
    cx, cy = hsd
    plt.scatter(x=cx.cpu().numpy(), y=cy.cpu().numpy(), **plot_kwargs)


def plot(cluster_centers: torch.Tensor,
         include_hsd: torch.Tensor,
         exclude_hsd: torch.Tensor = None,
         triangle: bool = True,
         plot_kwargs: dict = None):
    if not plot_kwargs:
        plot_kwargs = {}

    exclude_hsd_kwargs = plot_kwargs.get('exclude_hsd') or {}
    include_hsd_kwargs = plot_kwargs.get('include_hsd') or {}
    cluster_centers_kwargs = plot_kwargs.get('cluster_centers') or {}
    chromacity_triangle_kwargs = plot_kwargs.get('chromacity_triangle') or {}

    has_noise = torch.is_tensor(exclude_hsd)
    if has_noise:
        plot_exclude_hsd(exclude_hsd, **exclude_hsd_kwargs)
    plot_cluster_centers(cluster_centers, **cluster_centers_kwargs)
    plot_include_hsd(include_hsd, **include_hsd_kwargs)
    if triangle:
        plot_chromacity_triangle(**chromacity_triangle_kwargs)
    plt.title('Unmixed Stain Cluster Centers (k=%s)' % cluster_centers.shape[0])
