"""
train_cpu.py — CPU-friendly NCA training.

Same paper, same architecture, smaller everything so a laptop or free
Colab CPU runner can finish in a reasonable time.

Targets a single emoji per run. Budget ~2-4 hours per target on a modern
laptop CPU, ~6-8 hours on Colab free CPU.

Usage:
    python train_cpu.py --target heart
    python train_cpu.py --target seedling --steps 5000
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
# Reduced-spec config (vs. train.py)
#   - grid 32×32 instead of 40×40            (~36% fewer pixels)
#   - 12 channels instead of 16              (~25% smaller)
#   - 64 hidden units instead of 128         (4× fewer params in update net)
#   - pool 256 instead of 1024               (4× less memory)
#   - 4000 steps instead of 8000             (still converges on simple shapes)
#   - batch 4 instead of 8                   (faster per-step on CPU)
# Net: ~10× faster training, organism quality only mildly degraded.
# ──────────────────────────────────────────────────────────────────────────────

CHANNEL_N   = 12
HIDDEN      = 64
GRID_SIZE   = 32
POOL_SIZE   = 256
BATCH_SIZE  = 4
LR          = 2e-3
FIRE_RATE   = 0.5
WEIGHTS_DIR = Path('weights')
WEIGHTS_DIR.mkdir(exist_ok=True)

EMOJI_CODEPOINTS = {
    'lizard':   '1f98e',
    'heart':    '2764',
    'seedling': '1f331',
    'fish':     '1f420',
    'star':     '2b50',
}

# Recommend training only the most visually impactful targets on CPU.
# Heart for emotional resonance, lizard for canonical paper homage.
CPU_RECOMMENDED = ['heart', 'lizard']


def load_emoji_target(name: str, size: int = GRID_SIZE) -> torch.Tensor:
    code = EMOJI_CODEPOINTS[name]
    url = f'https://github.com/googlefonts/noto-emoji/raw/main/png/128/emoji_u{code}.png'
    img = Image.open(BytesIO(requests.get(url, timeout=30).content)).convert('RGBA')
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr[..., :3] *= arr[..., 3:4]
    return torch.from_numpy(arr).permute(2, 0, 1)


class NCA(nn.Module):
    def __init__(self, n_channels=CHANNEL_N, hidden=HIDDEN, fire_rate=FIRE_RATE):
        super().__init__()
        self.n_channels = n_channels
        self.fire_rate  = fire_rate

        identity = torch.tensor([[0,0,0],[0,1,0],[0,0,0]], dtype=torch.float32)
        sobel_x  = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)/8.
        sobel_y  = sobel_x.T.contiguous()
        kernels  = torch.stack([identity, sobel_x, sobel_y])
        kernels  = kernels.unsqueeze(1).repeat(n_channels, 1, 1, 1)
        kernels  = kernels.view(n_channels, 3, 1, 3, 3).reshape(n_channels * 3, 1, 3, 3)
        self.register_buffer('perceive_kernels', kernels)

        self.update_net = nn.Sequential(
            nn.Conv2d(n_channels * 3, hidden, 1),
            nn.ReLU(),
            nn.Conv2d(hidden, n_channels, 1, bias=False),
        )
        nn.init.zeros_(self.update_net[-1].weight)

    def _perceive(self, x):
        return F.conv2d(x, self.perceive_kernels, padding=1, groups=self.n_channels)

    @staticmethod
    def _alive(x):
        return F.max_pool2d(x[:, 3:4], 3, 1, 1) > 0.1

    def forward(self, x, steps=1):
        for _ in range(steps):
            pre  = self._alive(x)
            dx   = self.update_net(self._perceive(x))
            mask = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3]) < self.fire_rate).float()
            x    = x + dx * mask
            x    = x * (pre & self._alive(x)).float()
        return x


def make_seed(batch=1):
    x = torch.zeros(batch, CHANNEL_N, GRID_SIZE, GRID_SIZE)
    x[:, 3:, GRID_SIZE // 2, GRID_SIZE // 2] = 1.0
    return x


def random_circle_damage(x, radius_frac=0.3):
    B, _, H, W = x.shape
    for b in range(B):
        cx = np.random.randint(H); cy = np.random.randint(W)
        r  = int(max(H, W) * radius_frac * np.random.uniform(0.5, 1.0))
        yy, xx = np.ogrid[:H, :W]
        x[b, :, (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = 0.0
    return x


def train(target_name, n_steps=4000):
    print(f'\n═══ CPU-train NCA → {target_name} ({n_steps} steps) ═══')
    target = load_emoji_target(target_name)
    np.save(WEIGHTS_DIR / f'{target_name}_target.npy', target.numpy())

    model = NCA()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=1e-5)
    pool = make_seed(batch=POOL_SIZE)

    import time; t0 = time.time()
    for step in range(n_steps):
        idx = np.random.choice(POOL_SIZE, BATCH_SIZE, replace=False)
        x = pool[idx]

        # Damage 1 of 4 batch samples once training is past initial phase
        if step > 300:
            x[:1] = random_circle_damage(x[:1].clone())

        T = np.random.randint(48, 73)        # shorter than GPU version
        y = model(x, steps=T)
        loss = F.mse_loss(y[:, :4], target.unsqueeze(0).expand(BATCH_SIZE, -1, -1, -1))

        opt.zero_grad()
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                p.grad /= (p.grad.norm() + 1e-8)
        opt.step()
        sched.step()

        per = ((y[:, :4] - target.unsqueeze(0)) ** 2).mean(dim=(1, 2, 3))
        worst = per.argmax().item()
        with torch.no_grad():
            updated = y.detach()
            updated[worst] = make_seed(1)[0]
            pool[idx] = updated

        if step % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed / max(step, 1) * (n_steps - step)
            print(f'  step {step:5d}  loss {loss.item():.4f}  '
                  f'elapsed {elapsed/60:.1f}m  eta {eta/60:.1f}m')

    torch.save({'state_dict': model.state_dict(),
                'config': {'n_channels': CHANNEL_N,
                           'hidden': HIDDEN,
                           'grid_size': GRID_SIZE},
                'target_name': target_name},
               WEIGHTS_DIR / f'{target_name}.pt')
    print(f'  ✓ saved weights/{target_name}.pt  '
          f'(total time: {(time.time()-t0)/60:.1f}m)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='heart',
                        choices=list(EMOJI_CODEPOINTS) + ['recommended'])
    parser.add_argument('--steps', type=int, default=4000)
    args = parser.parse_args()

    targets = CPU_RECOMMENDED if args.target == 'recommended' else [args.target]
    for t in targets:
        train(t, n_steps=args.steps)

    print('\nDone. Upload weights/ to your molab notebook.')
    print('Tip: 2 well-trained targets > 5 mediocre ones. Quality of '
          'healing demo > number of options.')
