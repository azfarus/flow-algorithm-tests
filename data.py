"""MNIST data loading (seeded, normalized to [-1, 1])."""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_loaders(cfg):
    """Returns (train_loader, test_loader). Train shuffling is seeded so both
    methods see the exact same batches in the exact same order."""
    tfm = transforms.Compose([
        transforms.ToTensor(),                    # [0, 1]
        transforms.Normalize((0.5,), (0.5,)),     # [-1, 1]
    ])
    train_set = datasets.MNIST(cfg.data_root, train=True, download=True, transform=tfm)
    test_set = datasets.MNIST(cfg.data_root, train=False, download=True, transform=tfm)

    g = torch.Generator()
    g.manual_seed(cfg.seed)

    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
        num_workers=cfg.num_workers, pin_memory=True, generator=g,
    )
    test_loader = DataLoader(
        test_set, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    return train_loader, test_loader


@torch.no_grad()
def get_train_tensor(train_loader):
    """All training images as one (N, 1, 28, 28) tensor in [-1, 1], with the
    same normalization as the loader (used for wfmeanflow's forcing batch)."""
    return (train_loader.dataset.data.float().div(255.0).unsqueeze(1) - 0.5) / 0.5


@torch.no_grad()
def get_real_images(cfg, n, seed=1234, split="test"):
    """Deterministic batch of `n` real MNIST images in [-1, 1] (for eval)."""
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    ds = datasets.MNIST(cfg.data_root, train=(split == "train"), download=True, transform=tfm)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n]
    return torch.stack([ds[int(i)][0] for i in idx])
