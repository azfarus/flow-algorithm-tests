"""Mean Flow (Geng et al., 2025: "Mean Flows for One-step Generative Modeling").

The model learns the *average* velocity u(z_t, r, t) = 1/(t-r) * integral_r^t v(z_s, s) ds,
which satisfies the identity:

    u(z_t, r, t) = v(z_t, t) - (t - r) * d/dt u(z_t, r, t)

The total derivative d/dt u is computed with a Jacobian-vector product (JVP)
along the tangent (v, dr=0, dt=1):

    du/dt = v . d_z u + d_t u

Training target (stop-gradient):  u_tgt = sg[ v - (t - r) * du/dt ]
Loss: adaptive-weighted MSE,      w * ||u_theta - u_tgt||^2,  w = 1/sg(||err||^2 + c)

Sampling is ONE step:  x0 = z1 - u(z1, r=0, t=1)   (multi-step also supported).
"""
import torch
from torch.func import jvp


def sample_r_t(batch, device, ratio_r_eq_t):
    """Sample (r, t) with r <= t; a fraction of pairs has r == t."""
    r = torch.rand(batch, device=device)
    t = torch.rand(batch, device=device)
    r, t = torch.minimum(r, t), torch.maximum(r, t)
    eq = torch.rand(batch, device=device) < ratio_r_eq_t
    r = torch.where(eq, t, r)
    return r, t


def mf_loss(model, x0, cfg):
    """Returns (weighted_loss, metrics_dict). Gradients flow through u only."""
    b = x0.shape[0]
    r, t = sample_r_t(b, x0.device, cfg.mf_ratio_r_eq_t)
    eps = torch.randn_like(x0)
    tv = t.view(-1, 1, 1, 1)
    z_t = (1.0 - tv) * x0 + tv * eps
    v = eps - x0

    def u_fn(z_, r_, t_):
        return model(z_, t_, r_)

    # JVP: primals (z_t, r, t), tangents (v, 0, 1)  ->  (u, du/dt)
    u, dudt = jvp(u_fn, (z_t, r, t), (v, torch.zeros_like(r), torch.ones_like(t)))

    dt = (t - r).view(-1, 1, 1, 1)
    u_tgt = v - dt * dudt
    err = u - u_tgt.detach()                     # stop-gradient on the target
    per_sample = err.pow(2).mean(dim=(1, 2, 3))
    raw_mse = per_sample.mean()
    w = 1.0 / (per_sample.detach() + cfg.mf_adaptive_c)   # adaptive weighting
    loss = (w * per_sample).mean()
    return loss, {"loss": float(loss.detach()), "raw_mse": float(raw_mse.detach())}


@torch.no_grad()
def mf_sample(model, shape, steps, device, noise=None):
    """steps=1 is the classic one-step generation: x0 = z1 - u(z1, 0, 1).
    steps>1 splits [0, 1] into intervals and applies x_r = x_t - (t-r) u(x, r, t)."""
    x = noise if noise is not None else torch.randn(shape, device=device)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    for i in range(steps):
        t_cur, r_cur = ts[i], ts[i + 1]
        tt = torch.full((shape[0],), float(t_cur), device=device)
        rr = torch.full((shape[0],), float(r_cur), device=device)
        u = model(x, tt, rr)
        x = x - (t_cur - r_cur) * u
    return x.clamp(-1.0, 1.0)
