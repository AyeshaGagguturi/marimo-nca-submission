"""Save every frame of heart growth so we can pick the best-looking step."""
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

# Reuse the NCA class from train_cpu
sys.path.insert(0, str(Path(__file__).parent))
from train_cpu import NCA, make_seed, GRID_SIZE

TARGET = sys.argv[1] if len(sys.argv) > 1 else 'heart'
OUT = Path('frames_' + TARGET)
OUT.mkdir(exist_ok=True)

ckpt = torch.load(f'weights/{TARGET}.pt', map_location='cpu', weights_only=True)
m = NCA()
m.load_state_dict(ckpt['state_dict'])
m.eval()

x = make_seed()
with torch.no_grad():
    for step in range(101):
        rgba = x[0, :4].detach().numpy().transpose(1, 2, 0)
        rgba = np.clip(rgba, 0, 1)
        img = (rgba * 255).astype(np.uint8)
        pil = Image.fromarray(img, 'RGBA')
        pil = pil.resize((GRID_SIZE * 12, GRID_SIZE * 12), Image.NEAREST)
        bg = Image.new('RGBA', pil.size, (10, 12, 18, 255))
        bg.alpha_composite(pil)
        bg.convert('RGB').save(OUT / f'step_{step:03d}.png')
        x = m(x, steps=1)

print(f'saved 101 frames to {OUT}/')
print(f'open {OUT}/ and pick the best frame — that number is your grow_steps')
