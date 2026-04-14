"""
LeJEPA training script supporting three variants:

  baseline         — original: SIGReg + inv_loss both on projector outputs
  sigreg_on_emb    — Proposal 1 (issue #17): SIGReg on backbone embeddings,
                     inv_loss still on projector outputs
  local_proj       — Proposal 2 (issue #17): projector applied only to local
                     views and acts as a JEPA-style predictor; SIGReg on global
                     embeddings, inv_loss between local projections and global
                     embedding mean

Run examples:
  python train.py +variant=baseline    +lamb=0.02 +V=4 +proj_dim=16 +lr=2e-3 +bs=256 +epochs=200
  python train.py +variant=sigreg_on_emb +lamb=0.02 +V=4 +proj_dim=16 +lr=2e-3 +bs=256 +epochs=200
  python train.py +variant=local_proj  +lamb=0.02 +V_global=2 +V_local=2 +lr=2e-3 +bs=256 +epochs=200

Override the W&B project name:
  python train.py +variant=baseline ... +wandb_project=my-project

For local_proj the projector dim is fixed to match the embedding dim (512) so
that local projections live in the same space as global embeddings.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import timm
import wandb
import hydra
import tqdm
from omegaconf import DictConfig
from datasets import load_dataset
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torchvision.ops import MLP


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# SIGReg
# ---------------------------------------------------------------------------

class SIGReg(torch.nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, x):
        """x: (N, D) — any tensor with samples on dim 0."""
        # Cast to float32 for numerical stability; the buffers are already float32
        x = x.float()
        A = torch.randn(x.size(-1), 256, device=x.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (x @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        return (err @ self.weights) * x.size(-2)


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

EMB_DIM = 512


class BaselineEncoder(nn.Module):
    """Original architecture: backbone + MLP projector applied to all views."""

    def __init__(self, proj_dim=16):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch8_224",
            pretrained=False,
            num_classes=EMB_DIM,
            drop_path_rate=0.1,
            img_size=128,
        )
        self.proj = MLP(EMB_DIM, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x):
        """x: (N, V, C, H, W)"""
        N, V = x.shape[:2]
        emb = self.backbone(x.flatten(0, 1))                              # (N*V, EMB_DIM)
        proj = self.proj(emb).reshape(N, V, -1).transpose(0, 1)          # (V, N, proj_dim)
        return emb, proj


class LocalProjEncoder(nn.Module):
    """Proposal 2: projector only on local views, dim matches embedding dim."""

    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch8_224",
            pretrained=False,
            num_classes=EMB_DIM,
            drop_path_rate=0.1,
            img_size=128,
        )
        # Output dim == EMB_DIM so projections and embeddings share the same space
        self.proj = MLP(EMB_DIM, [2048, 2048, EMB_DIM], norm_layer=nn.BatchNorm1d)

    def forward_global(self, x):
        """x: (N, V_g, C, H, W) → (emb: N*V_g × D, N, V_g)"""
        N, V = x.shape[:2]
        return self.backbone(x.flatten(0, 1)), N, V

    def forward_local(self, x):
        """x: (N, V_l, C, H, W) → (proj: N*V_l × D, N, V_l)"""
        N, V = x.shape[:2]
        emb = self.backbone(x.flatten(0, 1))
        return self.proj(emb), N, V


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def _base_aug(crop_size, crop_scale):
    return v2.Compose([
        v2.RandomResizedCrop(crop_size, scale=crop_scale),
        v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        v2.RandomGrayscale(p=0.2),
        v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))]),
        v2.RandomApply([v2.RandomSolarize(threshold=128)], p=0.2),
        v2.RandomHorizontalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _test_transform():
    return v2.Compose([
        v2.Resize(128),
        v2.CenterCrop(128),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class UniformViewDataset(torch.utils.data.Dataset):
    """Used for baseline and sigreg_on_emb: all V views use the same augmentation."""

    def __init__(self, split, V=4):
        self.V = V
        self.ds = load_dataset("frgfm/imagenette", "160px", split=split)
        self.aug = _base_aug(128, (0.08, 1.0))
        self.test = _test_transform()

    def __getitem__(self, i):
        item = self.ds[i]
        img = item["image"].convert("RGB")
        transform = self.aug if self.V > 1 else self.test
        return torch.stack([transform(img) for _ in range(self.V)]), item["label"]

    def __len__(self):
        return len(self.ds)


class MultiCropDataset(torch.utils.data.Dataset):
    """Used for local_proj: returns global and local views separately."""

    def __init__(self, split, V_global=2, V_local=2):
        self.V_global = V_global
        self.V_local = V_local
        self.ds = load_dataset("frgfm/imagenette", "160px", split=split)
        self.global_aug = _base_aug(128, (0.3, 1.0))
        self.local_aug = _base_aug(96, (0.05, 0.3))
        self.test = _test_transform()

    def __getitem__(self, i):
        item = self.ds[i]
        img = item["image"].convert("RGB")
        global_views = torch.stack([self.global_aug(img) for _ in range(self.V_global)])
        local_views = torch.stack([self.local_aug(img) for _ in range(self.V_local)])
        return global_views, local_views, item["label"]

    def __len__(self):
        return len(self.ds)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@hydra.main(version_base=None)
def main(cfg: DictConfig):
    variant = cfg.get("variant", "baseline")
    assert variant in ("baseline", "sigreg_on_emb", "local_proj"), (
        f"Unknown variant '{variant}'. Choose: baseline, sigreg_on_emb, local_proj"
    )

    seed = cfg.get("seed", 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    wandb_project = cfg.get("wandb_project", "LeJEPA-projector-ablation")
    wandb.init(
        project=wandb_project,
        name=f"{variant}-seed{seed}",
        config=dict(cfg),
    )

    device = get_device()
    print(f"Using device: {device}")

    # On MPS/CPU, mixed precision uses float16; on CUDA we use bfloat16.
    # GradScaler is only useful for float16 (not bfloat16, not MPS).
    if device.type == "cuda":
        amp_dtype = torch.bfloat16
        use_scaler = False   # bfloat16 doesn't need loss scaling
    elif device.type == "mps":
        amp_dtype = torch.float16
        use_scaler = True
    else:
        amp_dtype = torch.float32
        use_scaler = False

    scaler = GradScaler(device=device.type, enabled=use_scaler)

    sigreg = SIGReg().to(device)
    probe = nn.Sequential(nn.LayerNorm(EMB_DIM), nn.Linear(EMB_DIM, 10)).to(device)

    # -----------------------------------------------------------------------
    if variant in ("baseline", "sigreg_on_emb"):
        proj_dim = cfg.get("proj_dim", 16)
        V = cfg.get("V", 4)
        net = BaselineEncoder(proj_dim=proj_dim).to(device)
        train_ds = UniformViewDataset("train", V=V)
        test_ds = UniformViewDataset("validation", V=1)

    else:  # local_proj
        V_global = cfg.get("V_global", 2)
        V_local = cfg.get("V_local", 2)
        net = LocalProjEncoder().to(device)
        train_ds = MultiCropDataset("train", V_global=V_global, V_local=V_local)
        test_ds = UniformViewDataset("validation", V=1)

    # MPS doesn't support multiple DataLoader workers reliably
    num_workers = 0 if device.type == "mps" else 8

    train = DataLoader(train_ds, batch_size=cfg.bs, shuffle=True, drop_last=True, num_workers=num_workers)
    test = DataLoader(test_ds, batch_size=256, num_workers=num_workers)

    g1 = {"params": net.parameters(), "lr": cfg.lr, "weight_decay": 5e-2}
    g2 = {"params": probe.parameters(), "lr": 1e-3, "weight_decay": 1e-7}
    opt = torch.optim.AdamW([g1, g2])
    warmup_steps = len(train)
    total_steps = len(train) * cfg.epochs
    s1 = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=1e-3)
    scheduler = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])

    for epoch in range(cfg.epochs):
        net.train()
        probe.train()

        for batch in tqdm.tqdm(train, total=len(train)):
            with autocast(device.type, dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
                # ----------------------------------------------------------
                if variant == "baseline":
                    vs, y = batch
                    vs = vs.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    emb, proj = net(vs)
                    inv_loss = (proj.mean(0) - proj).square().mean()
                    sigreg_loss = sigreg(proj.flatten(0, 1))       # on projections

                elif variant == "sigreg_on_emb":
                    vs, y = batch
                    vs = vs.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    emb, proj = net(vs)
                    inv_loss = (proj.mean(0) - proj).square().mean()
                    sigreg_loss = sigreg(emb)                      # on embeddings

                else:  # local_proj
                    global_vs, local_vs, y = batch
                    global_vs = global_vs.to(device, non_blocking=True)
                    local_vs = local_vs.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)

                    global_emb, N, V_g = net.forward_global(global_vs)  # (N*V_g, D)
                    local_proj_out, _, V_l = net.forward_local(local_vs) # (N*V_l, D)

                    # SIGReg targets the global embedding distribution
                    sigreg_loss = sigreg(global_emb)

                    # Predictor loss: local projections should predict the mean
                    # global embedding for the same image (JEPA-style)
                    global_mean = global_emb.reshape(N, V_g, -1).mean(1)       # (N, D)
                    local_proj_r = local_proj_out.reshape(N, V_l, -1)           # (N, V_l, D)
                    inv_loss = (local_proj_r - global_mean.unsqueeze(1)).square().mean()

                    emb = global_emb  # probe uses global embeddings
                    V = V_g

                # ----------------------------------------------------------
                lejepa_loss = sigreg_loss * cfg.lamb + inv_loss * (1 - cfg.lamb)
                y_rep = y.repeat_interleave(V)
                probe_loss = F.cross_entropy(probe(emb.detach()), y_rep)
                loss = lejepa_loss + probe_loss

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            wandb.log({
                "train/probe": probe_loss.item(),
                "train/lejepa": lejepa_loss.item(),
                "train/sigreg": sigreg_loss.item(),
                "train/inv": inv_loss.item(),
            })

        # Evaluation
        net.eval()
        probe.eval()
        correct = 0
        with torch.inference_mode():
            for vs, y in test:
                vs = vs.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with autocast(device.type, dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
                    if variant == "local_proj":
                        emb, _, _ = net.forward_global(vs)
                    else:
                        emb, _ = net(vs)
                    correct += (probe(emb).argmax(1) == y).sum().item()
        wandb.log({"test/acc": correct / len(test_ds), "test/epoch": epoch})

    wandb.finish()


if __name__ == "__main__":
    main()
