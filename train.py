"""
train.py — Train Neural Cellular Automata on target emojis.
Run this on Nebius GPU once. The marimo notebook loads the resulting weights.

Usage:
    python train.py --target heart --steps 8000
    python train.py --target all  # trains all 5 targets sequentially

Produces:
    weights/{target}.pt         — trained NCA weights
    weights/{target}_target.npy — 40x40 RGBA target image
"""

import argparse
import os
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────────────

CHANNEL_N      = 16      # 3 RGB + 1 alpha + 12 hidden "thought" channels
TARGET_SIZE    = 40      # grid is 40x40 (training & inference)
POOL_SIZE      = 1024    # pool-based training for robustness
BATCH_SIZE     = 8
LR             = 2e-3
FIRE_RATE      = 0.5     # probability a cell updates per step
DEVICE         = 'cuda' if torch.cuda.is_available() else 'cpu'
WEIGHTS_DIR    = Path('weights')
WEIGHTS_DIR.mkdir(exist_ok=True)

# Emojis carry meaning. Pick ones with that feel alive.
EMOJI_CODEPOINTS = {
    'lizard':   '1f98e',   # 🦎 the paper's canonical demo — pays homage
    'heart':    '2764',    # ❤️  the emotional anchor
    'seedling': '1f331',   # 🌱 growth as metaphor
    'fish':     '1f420',   # 🐠 color + movement
    'star':     '2b50',    # ⭐ geometric cleanliness
}

# ──────────────────────────────────────────────────────────────────────────────
#  Target image loading — grabs official Noto emoji, normalizes to RGBA 40×40
# ──────────────────────────────────────────────────────────────────────────────

def load_emoji_target(name: str) -> torch.Tensor:
    """Download an emoji and convert to a 40×40 RGBA tensor in [0,1]."""
    code = EMOJI_CODEPOINTS[name]
    url = f'https://github.com/googlefonts/noto-emoji/raw/main/png/128/emoji_u{code}.png'
    img_bytes = requests.get(url, timeout=30).content
    img = Image.open(BytesIO(img_bytes)).convert('RGBA')
    img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0       # (H, W, 4)
    # Premultiply alpha — visually correct compositing against black bg
    arr[..., :3] *= arr[..., 3:4]
    return torch.from_numpy(arr).permute(2, 0, 1)        # (4, H, W)


# ──────────────────────────────────────────────────────────────────────────────
#  The Neural Cellular Automaton
# ──────────────────────────────────────────────────────────────────────────────

class NCA(nn.Module):
    """
    A differentiable cellular automaton.
    Each cell holds a 16-dim state vector. At each step every cell:
      1. perceives a 3×3 neighborhood via fixed Sobel filters (identity + gradients)
      2. computes an update via a 2-layer MLP (implemented as 1×1 convs)
      3. stochastically applies that update (fire_rate)
      4. survives only if it or a neighbor is "alive" (alpha > 0.1)
    """

    def __init__(self, n_channels: int = CHANNEL_N, hidden: int = 128,
                 fire_rate: float = FIRE_RATE):
        super().__init__()
        self.n_channels = n_channels
        self.fire_rate  = fire_rate

        # Fixed perception filters: identity, Sobel-x, Sobel-y
        identity = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
        sobel_x  = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8.0
        sobel_y  = sobel_x.T.contiguous()
        kernels  = torch.stack([identity, sobel_x, sobel_y])              # (3, 3, 3)

        # Apply each of the 3 kernels to each of the n_channels → 3·n_channels output
        kernels = kernels.unsqueeze(1).repeat(n_channels, 1, 1, 1)        # (3·C, 1, 3, 3)
        # Rearrange so groups=n_channels in depthwise conv:
        # we want output channel k*3+j to come from input channel k with kernel j
        kernels = kernels.view(n_channels, 3, 1, 3, 3).permute(0, 1, 2, 3, 4)
        kernels = kernels.reshape(n_channels * 3, 1, 3, 3)
        self.register_buffer('perceive_kernels', kernels)

        self.update_net = nn.Sequential(
            nn.Conv2d(n_channels * 3, hidden, 1),
            nn.ReLU(),
            nn.Conv2d(hidden, n_channels, 1, bias=False),
        )
        # Critical: zero-init final layer so the first step is a near-no-op.
        # Without this, the grid explodes and training never recovers.
        nn.init.zeros_(self.update_net[-1].weight)

    def _perceive(self, x: torch.Tensor) -> torch.Tensor:
        # depthwise conv: each channel convolved with [identity, sobel_x, sobel_y]
        return F.conv2d(x, self.perceive_kernels, padding=1, groups=self.n_channels)

    @staticmethod
    def _alive(x: torch.Tensor) -> torch.Tensor:
        """A cell is alive iff any neighbor (incl. itself) has alpha > 0.1."""
        alpha = x[:, 3:4]
        return F.max_pool2d(alpha, kernel_size=3, stride=1, padding=1) > 0.1

    def forward(self, x: torch.Tensor, steps: int = 1,
                return_trajectory: bool = False) -> torch.Tensor:
        trajectory = [x] if return_trajectory else None
        for _ in range(steps):
            pre_alive = self._alive(x)
            dx = self.update_net(self._perceive(x))
            # Stochastic update mask (shared across channels, broadcast)
            mask = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3],
                               device=x.device) < self.fire_rate).float()
            x = x + dx * mask
            post_alive = self._alive(x)
            alive = (pre_alive & post_alive).float()
            x = x * alive
            if return_trajectory:
                trajectory.append(x)
        return (x, trajectory) if return_trajectory else x


# ──────────────────────────────────────────────────────────────────────────────
#  Seed + damage helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_seed(n_channels: int = CHANNEL_N, size: int = TARGET_SIZE,
              batch: int = 1, device: str = DEVICE) -> torch.Tensor:
    """A single live cell at the center — everything grows from here."""
    x = torch.zeros(batch, n_channels, size, size, device=device)
    x[:, 3:, size // 2, size // 2] = 1.0   # alpha + hidden channels = 1 at center
    return x


def random_damage(x: torch.Tensor, radius_frac: float = 0.3) -> torch.Tensor:
    """Zero out a random circular region — teaches the model to heal."""
    B, _, H, W = x.shape
    for b in range(B):
        cx = np.random.randint(H)
        cy = np.random.randint(W)
        r  = int(max(H, W) * radius_frac * np.random.uniform(0.5, 1.0))
        yy, xx = np.ogrid[:H, :W]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        x[b, :, mask] = 0.0
    return x


# ──────────────────────────────────────────────────────────────────────────────
#  Training loop — pool-based, with damage for healing robustness
# ──────────────────────────────────────────────────────────────────────────────

def train(target_name: str, n_steps: int = 8000, damage: bool = True):
    print(f'\n═══ Training NCA → {target_name} ({n_steps} steps, device={DEVICE}) ═══')

    target = load_emoji_target(target_name).to(DEVICE)               # (4, H, W)
    np.save(WEIGHTS_DIR / f'{target_name}_target.npy',
            target.cpu().numpy())

    model = NCA().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=1e-5)

    # Pool of persistent states. Each training iter we take 8, advance them,
    # compute loss, backprop, and put them back. The one with highest loss is
    # replaced with a fresh seed each iter — keeps distribution diverse.
    pool = make_seed(batch=POOL_SIZE, device=DEVICE)

    losses = []
    for step in range(n_steps):
        # Sample batch indices
        idx = np.random.choice(POOL_SIZE, BATCH_SIZE, replace=False)
        x = pool[idx]

        # Always reseed the worst performer (sorted descending below)
        # (placeholder — computed after forward)

        # Optionally apply damage to a couple of states (robustness to injury)
        if damage and step > 500:
            x[:2] = random_damage(x[:2].clone(), radius_frac=0.3)

        # Random number of forward steps — model must be robust to time
        T = np.random.randint(64, 97)
        y = model(x, steps=T)

        # Loss: MSE on RGBA channels vs target, averaged over batch
        loss = F.mse_loss(y[:, :4], target.unsqueeze(0).expand(BATCH_SIZE, -1, -1, -1))

        opt.zero_grad()
        loss.backward()
        # Normalize per-parameter gradients — essential stability trick from the paper
        for p in model.parameters():
            if p.grad is not None:
                p.grad /= (p.grad.norm() + 1e-8)
        opt.step()
        sched.step()

        # Per-sample loss, to decide which state to reseed
        per_sample = ((y[:, :4] - target.unsqueeze(0)) ** 2).mean(dim=(1, 2, 3))
        worst = per_sample.argmax().item()

        # Write results back to pool; reseed the worst sample
        with torch.no_grad():
            updated = y.detach()
            updated[worst] = make_seed(batch=1, device=DEVICE)[0]
            pool[idx] = updated

        losses.append(loss.item())
        if step % 200 == 0:
            print(f'  step {step:5d}  loss {loss.item():.4f}  lr {sched.get_last_lr()[0]:.2e}')

    torch.save({'state_dict': model.state_dict(),
                'target_name': target_name,
                'final_loss': float(np.mean(losses[-100:]))},
               WEIGHTS_DIR / f'{target_name}.pt')
    print(f'  ✓ saved weights/{target_name}.pt  (final loss {np.mean(losses[-100:]):.4f})')


# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='all',
                        choices=list(EMOJI_CODEPOINTS) + ['all'])
    parser.add_argument('--steps', type=int, default=8000)
    parser.add_argument('--no-damage', action='store_true',
                        help='Disable damage during training (faster but fragile models)')
    args = parser.parse_args()

    targets = list(EMOJI_CODEPOINTS) if args.target == 'all' else [args.target]
    for t in targets:
        train(t, n_steps=args.steps, damage=not args.no_damage)

    print('\nAll done. Transfer the weights/ folder to your molab notebook.')
