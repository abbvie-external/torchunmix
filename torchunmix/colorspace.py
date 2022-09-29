import torch


def _adjust(x): return torch.ones_like(x, device=x.device) * 1e-6


def rgb_to_od(rgb: torch.Tensor, background: int = 255) -> torch.Tensor:
    mask = (rgb == 0)
    rgb[mask] = 1
    od = torch.maximum(-1 * torch.log(rgb / background), _adjust(rgb))
    return od


def od_to_hsd(od: torch.Tensor) -> torch.Tensor:
    od_mu = torch.mean(od, dim=1)
    cx = (od[:, 0, ...] / od_mu) - 1
    cy = (od[:, 1, ...] - od[:, 2, ...]) / (od_mu * torch.sqrt(torch.tensor([3], device=od_mu.device)))
    return torch.stack([cx, cy, od_mu], dim=1)


def hsd_to_od(hsd: torch.Tensor) -> torch.Tensor:
    cx = hsd[:, 0, ...]
    cy = hsd[:, 1, ...]
    od_mu = hsd[:, 2, ...] if len(hsd.shape) == 3 else 1
    principal_sqrt3 = torch.sqrt(torch.tensor([3], device=cx.device))
    r_od = od_mu * (cx + 1)
    stack = r_od.shape
    if not stack:
        r_od = torch.tensor([r_od], device=cx.device)
    g_od = 0.5 * od_mu * (2 - cx + principal_sqrt3 * cy)
    b_od = 0.5 * od_mu * (2 - cx - principal_sqrt3 * cy)
    if not stack:
        return torch.tensor([r_od, g_od, b_od], device=cx.device)
    return torch.stack([r_od, g_od, b_od], dim=1)


def od_to_rgb(od: torch.Tensor) -> torch.Tensor:
    od = torch.maximum(od, _adjust(od))
    return (255 * torch.exp(-1 * od)).to(torch.uint8)


def rgb_to_stains(rgb: torch.Tensor, conv_matrix: torch.Tensor, background: int = 255) -> torch.Tensor:
    rgb = rgb.permute(0, 2, 3, 1)
    rgb = torch.maximum(rgb / background, _adjust(rgb))
    stains = (torch.log(rgb) / torch.log(_adjust(rgb))) @ conv_matrix
    stains = torch.maximum(stains, torch.zeros_like(stains, device=stains.device))
    return stains.permute(0, 3, 1, 2)


def stains_to_rgb(stains: torch.Tensor, conv_matrix: torch.Tensor) -> torch.Tensor:
    stains = stains.permute(0, 2, 3, 1)
    log_rgb = -(stains * -torch.log(_adjust(stains))) @ conv_matrix
    rgb = torch.exp(log_rgb)
    return torch.clip(rgb, 0, 1).permute(0, 3, 1, 2)
