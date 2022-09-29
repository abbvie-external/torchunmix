from .functional import augment_stains

from typing import Tuple, List, Optional
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import torch


Ranges = Tuple[Tuple[float, float], Tuple[float, float], Optional[Tuple[float, float]]]


def plot_ranges(rgb: torch.Tensor,
                mat_to_stain: torch.Tensor,
                mat_to_rgb: torch.Tensor,
                ranges: Ranges,
                steps: int,
                threshold: float = 0.03,
                labels: List[str] = None,
                fontsize: int = 12) -> Tuple[Figure, Axes]:

    nchannels = len(ranges)

    if labels:
        assert len(labels) == nchannels
    else:
        labels = ['Stain #%s' % (x + 1) for x in range(nchannels)]

    if rgb.ndim == 3:
        rgb = rgb.unsqueeze(0)

    fig, ax = plt.subplots(nrows=nchannels, ncols=steps)

    for y in range(nchannels):

        current_range = ranges[y]
        current_steps = torch.linspace(current_range[0], current_range[1], steps)
        mat_augment = torch.Tensor([[0.0, 0.0, 0.0]]).to(rgb.device)

        if labels:
            ax[y, 0].set_ylabel(labels[y], fontsize=fontsize)

        for x in range(steps):
            current_value = current_steps[x]
            mat_augment[0, y] = current_value
            img = augment_stains(rgb, mat_to_stain, mat_to_rgb, mat_augment, threshold=threshold)
            ax[y, x].imshow(img[0].permute(1, 2, 0).cpu().numpy())
            ax[y, x].set_xticks([])
            ax[y, x].set_yticks([])
            ax[y, x].set_xlabel(('{0:.4f}').format(current_value), fontsize=fontsize)

    fig.suptitle("Stain Augmentation Ranges", fontsize=fontsize)
    fig.tight_layout()
    return fig, ax


def plot_mixture(rgb: torch.Tensor,
                 mat_to_stain: torch.Tensor,
                 mat_to_rgb: torch.Tensor,
                 ranges: Tuple[Tuple[float, float], Tuple[float, float]],
                 steps: int,
                 channels: Tuple[int, int],
                 threshold: float = 0.03,
                 labels: List[str] = None,
                 fontsize: int = 12) -> Tuple[Figure, Axes]:

    if rgb.ndim == 3:
        rgb = rgb.unsqueeze(0)

    if labels:
        assert len(labels) == 2

    ranges_y = torch.linspace(ranges[0][1], ranges[0][0], steps)
    ranges_x = torch.linspace(ranges[1][0], ranges[1][1], steps)
    fig, ax = plt.subplots(nrows=steps, ncols=steps)
    fig.supylabel('Stain #%s' % (channels[0] + 1) if not labels else labels[0], fontsize=fontsize)
    fig.supxlabel('Stain #%s' % (channels[1] + 1) if not labels else labels[1], fontsize=fontsize)

    for y in range(steps):

        y_val = ranges_y[y].item()
        ax[y, 0].set_ylabel('{0:.4f}'.format(y_val), fontsize=fontsize)

        for x in range(steps):
            x_val = ranges_x[x].item()
            mat_augment = torch.Tensor([[0.0, 0.0, 0.0]]).to(rgb.device)
            mat_augment[0, channels[0]] = y_val
            mat_augment[0, channels[1]] = x_val
            img = augment_stains(rgb, mat_to_stain, mat_to_rgb, mat_augment, threshold=threshold)
            ax[y, x].imshow(img[0].permute(1, 2, 0).cpu().numpy())
            ax[y, x].set_xticks([])
            ax[y, x].set_yticks([])
            if y == steps - 1:
                ax[y, x].set_xlabel('{0:.4f}'.format(x_val), fontsize=fontsize)

    fig.suptitle("Stain Augmentation Mixture", fontsize=fontsize)
    fig.tight_layout()
    return fig, ax
