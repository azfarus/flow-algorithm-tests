"""Flow Matching (conditional flow matching with linear interpolation path).

Path:      x_t = (1 - t) * x0 + t * eps,   eps ~ N(0, I)
Velocity:  v   = eps - x0  (target)
Sampling:  integrate the learned ODE dx/dt = v_theta(x, t) from t=1 to t=0
           with Euler steps (multi-step generation).
"""
import torch
import torch.nn.functional as F


def fm_loss(model, x0):
    """Returns (loss, metrics_dict)."""
    b = x0.shape[0]
    t = torch.rand(b, device=x0.device)
    eps = torch.randn_like(x0)
    tv = t.view(-1, 1, 1, 1)
    x_t = (1.0 - tv) * x0 + tv * eps
    target = eps - x0
    pred = model(x_t, t)
    loss = F.mse_loss(pred, target)
    return loss, {"loss": float(loss.detach()), "raw_mse": float(loss.detach())}


@torch.no_grad()
def fm_sample(model, shape, steps, device, noise=None):
    """Euler integration from t=1 (noise) to t=0 (image) in `steps` NFEs."""
    x = noise if noise is not None else torch.randn(shape, device=device)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    for i in range(steps):
        t_cur = ts[i]
        dt = ts[i + 1] - ts[i]  # negative
        tt = torch.full((shape[0],), float(t_cur), device=device)
        v = model(x, tt)
        x = x + v * dt
    return x.clamp(-1.0, 1.0)
