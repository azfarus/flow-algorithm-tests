"""Weak-Form MeanFlow (see weak-form-meanflow/*.md): JVP-free, target-free MeanFlow.

Same average velocity u(z, r, t) and the same sampler as Mean Flow
(x0 = z1 - u(z1, 0, 1), mf_sample is reused). Only the training loss changes:
the MeanFlow residual R(u) = u - v + (t - r) D_t u is tested against random
smooth fields phi and integrated by parts, so every derivative lands on phi:

    W(phi; u) = A(phi; u) - b(phi)
    A = E_tri[ -1/2 (t-r) u(z_t,r,t) . (d_t phi + (v_c . grad_z) phi) ]
        + E[ (1-r') phi(z1,r',1) . u(z1,r',1) ],      z1 ~ N(0,I), r' ~ U[0,1]
    b = E_tri[ 1/2 phi(z_t,r,t) . v_c ],              v_c = e - x0

(r, t) uniform on the triangle r < t (density 2 -> factor 1/2). Loss per step:

    L = 1/K sum_k [ U-stat(A_k^2) - 2 mean(A_k) b_k + b_k^2 ] + lam_a * anchor

with the U-statistic over the main batch (unbiased at any batch size), b_k
from an independent auxiliary data batch (no network call), and the anchor =
plain flow matching on the diagonal r = t. Test fields (fresh every step):

    phi_k(z, r, t) = a_k eta_k(r, t) cos(omega_k . z + beta_k)

a_k unit vector (half along omega_k, half random), omega_k Gaussian on a ladder of scales, beta_k
uniform phase, eta_k a random cosine-basis polynomial in (r, t).
"""
import math

import torch


def sample_r_t_triangle(batch, device):
    """(r, t) uniform on {0 <= r < t <= 1} (density 2)."""
    r = torch.rand(batch, device=device)
    t = torch.rand(batch, device=device)
    return torch.minimum(r, t), torch.maximum(r, t)


def sample_test_fields(cfg, d, device):
    """K random Fourier test fields, shared by the whole batch."""
    K, L = cfg.wf_num_fields, cfg.wf_time_basis
    ladder = torch.tensor(cfg.wf_freq_scales, device=device, dtype=torch.float32)
    scale = ladder[torch.randint(len(ladder), (K,), device=device)]
    # omega . z has phase std ~ scale * (per-pixel rms of z)
    omega = torch.randn(K, d, device=device) * (scale / math.sqrt(d)).unsqueeze(1)
    # Output direction: half aligned with omega, half random. A purely random a
    # is ~orthogonal to omega at d=784 and the fields then see almost nothing
    # (measured: weak loss of u=0 and u* indistinguishable). E[a a^T] stays
    # isotropic and the random part keeps the kernel positive definite.
    g = torch.randn(K, d, device=device)
    a = omega / omega.norm(dim=1, keepdim=True) + g / g.norm(dim=1, keepdim=True)
    a = a / a.norm(dim=1, keepdim=True)
    beta = 2.0 * math.pi * torch.rand(K, device=device)
    coef = torch.randn(K, L, L, device=device) / L
    return a, omega, beta, coef


def time_factor(coef, r, t):
    """eta_k(r, t) = sum_pq c_kpq cos(p pi r) cos(q pi t) and d/dt eta. Shapes (B, K)."""
    freqs = math.pi * torch.arange(coef.shape[1], device=r.device, dtype=r.dtype)
    cr = torch.cos(r[:, None] * freqs)
    ct = torch.cos(t[:, None] * freqs)
    dct = -freqs * torch.sin(t[:, None] * freqs)
    eta = torch.einsum("bp,kpq,bq->bk", cr, coef, ct)
    deta_dt = torch.einsum("bp,kpq,bq->bk", cr, coef, dct)
    return eta, deta_dt


@torch.no_grad()
def forcing(fields, data_pool, cfg, device):
    """b_k = E[1/2 phi_k . v_c] from an independent auxiliary batch (no network)."""
    a, omega, beta, coef = fields
    n_aux = cfg.wf_aux_batch
    x = data_pool[torch.randint(len(data_pool), (n_aux,), device=data_pool.device)]
    x = x.to(device).flatten(1)
    e = torch.randn_like(x)
    r, t = sample_r_t_triangle(n_aux, device)
    z = (1.0 - t[:, None]) * x + t[:, None] * e
    v = e - x
    eta, _ = time_factor(coef, r, t)
    phi_dot_v = eta * torch.cos(z @ omega.T + beta) * (v @ a.T)
    return 0.5 * phi_dot_v.mean(0)


def wf_loss(model, x0, cfg, data_pool):
    """Returns (loss, metrics_dict). data_pool: all training images (N, C, H, W),
    used only for the auxiliary forcing batch. Gradients flow through plain
    forward evaluations of u (no JVP, no stop-gradient target)."""
    b = x0.shape[0]
    device = x0.device
    n_anchor = int(round(cfg.wf_anchor_ratio * b))
    m = b - n_anchor
    assert m >= 2, "weak term needs at least 2 samples for the U-statistic"
    d = x0[0].numel()

    # ---- main batch: triangle samples (weak term) + diagonal samples (anchor)
    r, t = sample_r_t_triangle(b, device)
    r[m:] = t[m:]
    eps = torch.randn_like(x0)
    tv = t.view(-1, 1, 1, 1)
    z_t = (1.0 - tv) * x0 + tv * eps
    v = eps - x0
    # boundary samples: exact noise z1 at an independent r'
    z1 = torch.randn_like(x0[:m])
    r1 = torch.rand(m, device=device)
    ones = torch.ones(m, device=device)

    # one batched forward: u(z_t, r, t) for all samples and u(z1, r', 1)
    u_all = model(torch.cat([z_t, z1]), torch.cat([t, ones]), torch.cat([r, r1]))
    u, u1 = u_all[:b], u_all[b:]

    # ---- weak residual, u-dependent part A_ik (i over m triangle samples)
    fields = sample_test_fields(cfg, d, device)
    a, omega, beta, coef = fields
    zf, vf, uf = z_t[:m].flatten(1), v[:m].flatten(1), u[:m].flatten(1)
    theta = zf @ omega.T + beta
    eta, deta_dt = time_factor(coef, r[:m], t[:m])
    # (d_t phi + (v_c . grad) phi) . u = (u . a) [d_t eta cos - eta sin (omega . v_c)]
    transport = (uf @ a.T) * (deta_dt * torch.cos(theta)
                              - eta * torch.sin(theta) * (vf @ omega.T))
    eta1, _ = time_factor(coef, r1, ones)
    boundary = eta1 * torch.cos(z1.flatten(1) @ omega.T + beta) * (u1.flatten(1) @ a.T)
    A = -0.5 * (t[:m] - r[:m]).unsqueeze(1) * transport + (1.0 - r1).unsqueeze(1) * boundary

    # ---- forcing b_k from an independent auxiliary batch
    bk = forcing(fields, data_pool, cfg, device)

    # ---- unbiased estimate of mean_k W_k^2 (U-statistic + decoupled forcing)
    s, q = A.sum(0), A.pow(2).sum(0)
    u_stat = (s.pow(2) - q) / (m * (m - 1))
    weak = (u_stat - 2.0 * A.mean(0) * bk + bk.pow(2)).mean()

    # ---- diagonal anchor: flow matching at r = t
    anchor = (u[m:] - v[m:]).pow(2).mean() if n_anchor > 0 else u.sum() * 0.0
    loss = weak + cfg.wf_anchor_weight * anchor
    return loss, {"loss": float(loss.detach()), "raw_mse": float(anchor.detach()),
                  "weak_loss": float(weak.detach()), "anchor_mse": float(anchor.detach())}
