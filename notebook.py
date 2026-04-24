"""
The Language of Life
A marimo notebook on Neural Cellular Automata (Mordvintsev et al., 2020).

This notebook does not try to be exhaustive. It tries to be moving.
It walks you from a single pixel through emergence, self-organization,
and healing — and asks, quietly, what life actually is.

Submission for the marimo × alphaXiv "Bring Research to Life" competition.
"""

import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium", app_title="The Language of Life")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from pathlib import Path
    from PIL import Image
    import io
    import base64

    return F, Image, Path, base64, io, mo, nn, np, torch


@app.cell
def _(mo):
    # Global style — dark, cinematic, calm.
    mo.Html("""
    <style>
      .mo-cell { background: transparent; }
      .prose {
        font-family: 'Georgia', 'Iowan Old Style', serif;
        font-size: 1.08rem;
        line-height: 1.75;
        color: #e8e8e8;
        max-width: 680px;
        margin: 0 auto;
      }
      .prose em { color: #ffd78a; font-style: italic; }
      .chapter {
        font-family: 'Inter', system-ui, sans-serif;
        font-size: 0.8rem;
        letter-spacing: 0.3em;
        text-transform: uppercase;
        color: #8aa4c8;
        margin-top: 4rem;
        margin-bottom: 0.5rem;
      }
      .h1 {
        font-family: 'Georgia', serif;
        font-size: 2.8rem;
        line-height: 1.1;
        color: #ffffff;
        margin-bottom: 2rem;
        font-weight: 400;
      }
      .caption {
        font-family: 'Inter', system-ui, sans-serif;
        font-size: 0.85rem;
        color: #8fa3bd;
        text-align: center;
        font-style: italic;
        margin-top: 0.5rem;
      }
    </style>
    """)
    return


@app.cell
def _(mo):
    mo.Html("""
    <div class="prose" style="text-align:center; margin-top: 2rem;">
      <div class="chapter">A notebook</div>
      <div class="h1">The Language of Life</div>
      <div style="color:#aab8cc; font-style:italic;">
        On Neural Cellular Automata &middot; after Mordvintsev et al.
      </div>
    </div>
    """)
    return


@app.cell
def _(mo):
    mo.Html("""
    <div class="prose" style="margin-top: 3rem;">
      <p>There's a single cell at the beginning of every story. Before the
      heartbeat. Before the first breath. Before anything you'd recognize as
      <em>alive</em>, there was just one.</p>

      <p>It divided. Its children divided. And somewhere in that quiet
      branching — without a blueprint, without a conductor, without anyone
      in charge — <em>you</em> happened.</p>

      <p>This is a notebook about the simplest possible version of that
      miracle.</p>
    </div>
    """)
    return


@app.cell
def _(F, Path, nn, np, torch):
    # Auto-detect training config from the first available checkpoint,
    # so the notebook works whether you trained at GPU scale (16/40/128)
    # or CPU scale (12/32/64).
    import torch as _torch
    _first_ckpt = next(Path('weights').glob('*.pt'), None)
    if _first_ckpt is not None:
        _cfg = _torch.load(_first_ckpt, map_location='cpu',
                           weights_only=True).get('config', {})
        CHANNEL_N = _cfg.get('n_channels', 16)
        GRID_SIZE = _cfg.get('grid_size', 40)
        HIDDEN    = _cfg.get('hidden', 128)
    else:
        CHANNEL_N, GRID_SIZE, HIDDEN = 16, 40, 128

    class NCA(nn.Module):
        """A cell is a 16-dim vector. The network is the laws of its physics."""

        def __init__(self, n_channels=CHANNEL_N, hidden=HIDDEN, fire_rate=0.5):
            super().__init__()
            self.n_channels = n_channels
            self.fire_rate  = fire_rate

            identity = torch.tensor([[0,0,0],[0,1,0],[0,0,0]], dtype=torch.float32)
            sobel_x  = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)/8.
            sobel_y  = sobel_x.T.contiguous()
            kernels  = torch.stack([identity, sobel_x, sobel_y])
            kernels  = kernels.unsqueeze(1).repeat(n_channels, 1, 1, 1)
            kernels  = kernels.view(n_channels, 3, 1, 3, 3)
            kernels  = kernels.reshape(n_channels * 3, 1, 3, 3)
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

        def forward(self, x, steps=1, return_trajectory=False, fire_rate=None):
            fr = self.fire_rate if fire_rate is None else fire_rate
            traj = [x.clone()] if return_trajectory else None
            for _ in range(steps):
                pre = self._alive(x)
                dx  = self.update_net(self._perceive(x))
                mask = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3],
                                   device=x.device) < fr).float()
                x = x + dx * mask
                post = self._alive(x)
                x = x * (pre & post).float()
                if return_trajectory:
                    traj.append(x.clone())
            return (x, traj) if return_trajectory else x

    def make_seed(n_ch=CHANNEL_N, size=GRID_SIZE, batch=1, device='cpu'):
        x = torch.zeros(batch, n_ch, size, size, device=device)
        x[:, 3:, size // 2, size // 2] = 1.0
        return x

    def to_rgba_img(x):
        """Convert a single NCA state (C,H,W) tensor → uint8 RGBA image array."""
        rgba = x[:4].detach().cpu().numpy().transpose(1, 2, 0)
        rgba = np.clip(rgba, 0, 1)
        return (rgba * 255).astype(np.uint8)


    return GRID_SIZE, NCA, make_seed, to_rgba_img


@app.cell
def _(NCA, Path, np, torch):
    WEIGHTS_DIR = Path('weights')
    AVAILABLE_TARGETS = []
    MODELS = {}
    TARGETS = {}

    for wpath in sorted(WEIGHTS_DIR.glob('*.pt')):
        name = wpath.stem
        ckpt = torch.load(wpath, map_location='cpu', weights_only=True)
        m = NCA()
        m.load_state_dict(ckpt['state_dict'])
        m.eval()
        MODELS[name] = m
        TARGETS[name] = np.load(WEIGHTS_DIR / f'{name}_target.npy')
        AVAILABLE_TARGETS.append(name)

    _status = f"loaded {len(AVAILABLE_TARGETS)} trained models: {', '.join(AVAILABLE_TARGETS)}"
    return AVAILABLE_TARGETS, MODELS


@app.cell
def _(Image, base64, io, np, to_rgba_img):
    def frames_to_gif_b64(frames, scale=10, duration=60, loop_delay=800):
        """Build a GIF from a list of (C,H,W) tensors; return base64 data-URL."""
        imgs = []
        for f in frames:
            arr = to_rgba_img(f[0])
            img = Image.fromarray(arr, 'RGBA')
            img = img.resize((arr.shape[1] * scale, arr.shape[0] * scale),
                             Image.NEAREST)
            # Composite on dark background for premultiplied alpha
            bg = Image.new('RGBA', img.size, (10, 12, 18, 255))
            bg.alpha_composite(img)
            imgs.append(bg.convert('P', palette=Image.ADAPTIVE, colors=128))

        # Last frame lingers slightly longer for breathing room
        durations = [duration] * (len(imgs) - 1) + [loop_delay]
        buf = io.BytesIO()
        imgs[0].save(buf, format='GIF', save_all=True, append_images=imgs[1:],
                     duration=durations, loop=0, disposal=2, optimize=True)
        data = base64.b64encode(buf.getvalue()).decode()
        return f'data:image/gif;base64,{data}'

    def render_side_by_side(rgb_frames, latent_frames, scale=8, duration=80):
        """Render two synchronized frame sequences side by side as one GIF."""
        imgs = []
        for rgb_f, lat_f in zip(rgb_frames, latent_frames):
            left  = to_rgba_img(rgb_f[0])
            # Visualize latent channel 4 (one of the "thought" channels)
            lat = lat_f[0, 4].detach().cpu().numpy()
            lat = (lat - lat.min()) / (lat.max() - lat.min() + 1e-8)
            # Colormap via simple viridis-like transform
            import matplotlib.cm as cm
            colored = (cm.inferno(lat) * 255).astype(np.uint8)

            left_img  = Image.fromarray(left, 'RGBA').resize(
                (left.shape[1] * scale, left.shape[0] * scale), Image.NEAREST)
            right_img = Image.fromarray(colored, 'RGBA').resize(
                (colored.shape[1] * scale, colored.shape[0] * scale), Image.NEAREST)

            gap = 16
            W = left_img.width + right_img.width + gap
            H = left_img.height
            canvas = Image.new('RGBA', (W, H), (10, 12, 18, 255))
            canvas.alpha_composite(left_img,  (0, 0))
            canvas.alpha_composite(right_img, (left_img.width + gap, 0))
            imgs.append(canvas.convert('P', palette=Image.ADAPTIVE, colors=128))

        durations = [duration] * len(imgs)
        durations[-1] = 1200
        buf = io.BytesIO()
        imgs[0].save(buf, format='GIF', save_all=True, append_images=imgs[1:],
                     duration=durations, loop=0, disposal=2, optimize=True)
        return 'data:image/gif;base64,' + base64.b64encode(buf.getvalue()).decode()

    def img_tag(src, caption=None, max_width=520):
        caption_html = f'<div class="caption">{caption}</div>' if caption else ''
        return (f'<div style="display:flex;flex-direction:column;align-items:center;'
                f'justify-content:center;width:100%;margin:2rem 0;">'
                f'<img src="{src}" style="max-width:{max_width}px;width:100%;'
                f'border-radius:8px;box-shadow:0 12px 48px rgba(0,0,0,.5);"/>'
                f'{caption_html}</div>')

    return frames_to_gif_b64, img_tag, render_side_by_side


@app.cell
def _(mo):
    mo.Html('<div class="prose"><div class="chapter">Chapter I</div>'
            '<div class="h1" style="font-size:2rem;">A single cell</div>'
            '<p>This is where we begin. One pixel, alive in the middle of '
            'nothing.</p></div>')
    return


@app.cell
def _(GRID_SIZE, Image, base64, img_tag, io, mo, np):
    # Hand-build the opening frame: a single bright dot in the void
    _arr = np.zeros((GRID_SIZE, GRID_SIZE, 4), dtype=np.uint8)
    _arr[GRID_SIZE // 2, GRID_SIZE // 2] = [255, 255, 255, 255]
    _img = Image.fromarray(_arr, 'RGBA').resize((400, 400), Image.NEAREST)
    _bg = Image.new('RGBA', _img.size, (10, 12, 18, 255))
    _bg.alpha_composite(_img)
    _buf = io.BytesIO()
    _bg.save(_buf, format='PNG')
    _src = 'data:image/png;base64,' + base64.b64encode(_buf.getvalue()).decode()
    mo.Html(img_tag(_src, caption='a seed — one cell in a 40×40 void'))
    return


@app.cell
def _(mo):
    mo.Html("""
    <div class="prose">
      <p>The dot is a vector: sixteen numbers. Three of them will become color.
      One is alpha — a commitment to exist. The other twelve are hidden
      state: a little private world the cell can use to think.</p>
      <p>It has no memory of what it's supposed to become.</p>
      <p>It has no neighbors yet. Just itself, and the rules.</p>
    </div>
    """)
    return


@app.cell
def _(mo):
    mo.Html('<div class="prose"><div class="chapter">Chapter II</div>'
            '<div class="h1" style="font-size:2rem;">Rules without rulers</div>'
            '<p>In 1970, a mathematician named John Conway published a game with '
            'three rules. A cell lives if two or three neighbors are alive. '
            'Otherwise it dies. If exactly three live neighbors surround a dead '
            'cell, a new one is born.</p>'
            '<p>That is the whole game. No goal. No plan. No one in charge.</p>'
            '<p>From those three rules came <em>gliders</em>. Spaceships. Guns '
            'that fire other patterns. Machines that compute. A whole universe, '
            'from a world that cannot count past four.</p>'
            '<p>The rules were fixed. What if they weren\'t?</p></div>')
    return


@app.cell
def _(Image, base64, img_tag, io, mo, np):
    # Tiny Conway GoL simulation → GIF, as a warm-up example
    def _gol_step(g):
        n = sum(np.roll(np.roll(g, i, 0), j, 1)
                for i in (-1, 0, 1) for j in (-1, 0, 1)) - g
        return ((n == 3) | ((g == 1) & (n == 2))).astype(np.uint8)

    _rng = np.random.default_rng(7)
    _g = _rng.integers(0, 2, size=(40, 40), dtype=np.uint8)

    _frames = []
    for _ in range(60):
        _arr = np.zeros((40, 40, 4), dtype=np.uint8)
        _arr[_g == 1] = [230, 230, 240, 255]
        _img = Image.fromarray(_arr, 'RGBA').resize((400, 400), Image.NEAREST)
        _bg = Image.new('RGBA', _img.size, (10, 12, 18, 255))
        _bg.alpha_composite(_img)
        _frames.append(_bg.convert('P', palette=Image.ADAPTIVE, colors=64))
        _g = _gol_step(_g)

    _buf = io.BytesIO()
    _frames[0].save(_buf, format='GIF', save_all=True, append_images=_frames[1:],
                    duration=120, loop=0, disposal=2, optimize=True)
    _src = 'data:image/gif;base64,' + base64.b64encode(_buf.getvalue()).decode()
    mo.Html(img_tag(_src, caption='Conway\'s Game of Life · three rules, no designer'))
    return


@app.cell
def _(mo):
    mo.Html('<div class="prose"><div class="chapter">Chapter III</div>'
            '<div class="h1" style="font-size:2rem;">What if the rules could learn?</div>'
            '<p>A <em>neural cellular automaton</em> keeps Conway\'s premise — '
            'cells only talk to neighbors, nobody is in charge — but its update '
            'rule is a small neural network. You show it a target shape. You '
            'say: grow into this. It figures out, on its own, how local neighbors '
            'should whisper to each other to make that shape appear from a single '
            'seed.</p>'
            '<p>Here is an <em>untrained</em> network doing its best.</p></div>')
    return


@app.cell
def _(NCA, frames_to_gif_b64, img_tag, make_seed, mo, torch):
    # Untrained NCA — should produce beautiful nonsense
    torch.manual_seed(0)
    _m = NCA()
    torch.nn.init.normal_(_m.update_net[-1].weight, std=1.0)
    # Bypass alive gating so chaotic updates can spread freely
    _m._alive = lambda x: torch.ones(x.shape[0], 1, x.shape[2], x.shape[3], dtype=torch.bool)
    _m.eval()
    # Start with a fully-alive grid so there's something to see
    _x = make_seed()
    _x[:, 3, :, :] = 1.0  # set all alpha=1 so every cell is alive from step 0
    with torch.no_grad():
        _, _traj = _m(_x, steps=80, return_trajectory=True)
    _traj_sub = _traj[::2]  # keep gif small
    _src = frames_to_gif_b64(_traj_sub, scale=10, duration=70)
    mo.Html(img_tag(_src, caption='untrained: rules chosen at random. '
                                  'no direction, no form.'))
    return


@app.cell
def _(mo):
    mo.Html('<div class="prose">'
            '<p>Noise. Beautiful noise, maybe, but noise.</p>'
            '<p>Now watch what training does.</p></div>')
    return


@app.cell
def _(
    AVAILABLE_TARGETS,
    MODELS,
    frames_to_gif_b64,
    img_tag,
    make_seed,
    mo,
    torch,
):
    # First trained model — the "reveal" moment
    _target = 'lizard' if 'lizard' in AVAILABLE_TARGETS else AVAILABLE_TARGETS[0]
    _m = MODELS[_target]
    _x = make_seed()
    with torch.no_grad():
        _, _traj = _m(_x, steps=96, return_trajectory=True)
    _src = frames_to_gif_b64(_traj[::3], scale=10, duration=70)
    mo.Html(img_tag(_src, caption=f'trained: one seed → an entire {_target}. '
                                   '~250,000 parameters, 96 steps.'))
    return


@app.cell
def _(mo):
    mo.Html("""
    <div class="prose">
      <p>The network doesn't contain a picture. It contains a <em>field of
      instincts</em> — a very small function that, repeated everywhere,
      eventually agrees on a shape.</p>
      <p>Every cell is running the same program. Every cell is only looking at
      its immediate neighbors. And yet a whole body emerges. This is the
      paper's central claim, and it is the real reason to care:</p>
      <p style="text-align:center; font-style:italic; color:#ffd78a;
                font-size:1.15rem; margin: 1.5rem 0;">
        Global form can arise from purely local communication.</p>
    </div>
    """)
    return


@app.cell
def _(mo):
    mo.Html('<div class="prose"><div class="chapter">Chapter IV</div>'
            '<div class="h1" style="font-size:2rem;">Becoming</div>'
            '<p>Pick a target. Each of these was trained from scratch — same '
            'architecture, same 16 channels, but a different destiny folded '
            'into 250,000 numbers.</p></div>')
    return


@app.cell
def _(AVAILABLE_TARGETS, mo):
    target_picker = mo.ui.dropdown(
        options=AVAILABLE_TARGETS,
        value=AVAILABLE_TARGETS[0],
        label='choose a target',
    )
    steps_slider = mo.ui.slider(start=16, stop=160, step=8, value=96,
                                label='growth steps', show_value=True)
    mo.hstack([target_picker, steps_slider], justify='start', gap=2)
    return steps_slider, target_picker


@app.cell
def _(
    MODELS,
    frames_to_gif_b64,
    img_tag,
    make_seed,
    mo,
    steps_slider,
    target_picker,
    torch,
):
    _m = MODELS[target_picker.value]
    _x = make_seed()
    with torch.no_grad():
        _, _traj = _m(_x, steps=int(steps_slider.value),
                      return_trajectory=True)
    _sub = _traj[::max(1, len(_traj) // 40)]
    _src = frames_to_gif_b64(_sub, scale=10, duration=70)
    mo.Html(img_tag(_src,
        caption=f'{target_picker.value} · {steps_slider.value} steps'))
    return


@app.cell
def _(mo):
    mo.Html("""
    <div class="prose">
      <div class="chapter">Chapter V</div>
      <div class="h1" style="font-size:2rem;">It remembers what it's supposed to be</div>
      <p>A single cell already has the whole organism folded inside it.</p>
      <p>That is a hard sentence to parse. Here is what it means: the rules
      don't <em>know</em> the shape. The rules only know how to ask your
      neighbors what they're doing. And yet, somehow, the shape always
      comes back.</p>
      <p>Let me show you what I mean.</p>
    </div>
    """)
    return


@app.cell
def _(AVAILABLE_TARGETS, mo):
    heal_target = mo.ui.dropdown(
        options=AVAILABLE_TARGETS,
        value='heart' if 'heart' in AVAILABLE_TARGETS else AVAILABLE_TARGETS[0],
        label='organism',
    )
    damage_type = mo.ui.dropdown(
        options=['hole (center)', 'cut in half', 'scatter damage', 'erase top'],
        value='cut in half',
        label='injury',
    )
    heal_btn = mo.ui.run_button(label='injure & heal')
    mo.hstack([heal_target, damage_type, heal_btn], justify='start', gap=2)
    return damage_type, heal_btn, heal_target


@app.cell
def _(
    GRID_SIZE,
    MODELS,
    damage_type,
    frames_to_gif_b64,
    heal_btn,
    heal_target,
    img_tag,
    make_seed,
    mo,
    np,
    torch,
):
    mo.stop(not heal_btn.value, mo.md(''))

    _m = MODELS[heal_target.value]

    # First: grow the organism to maturity
    _x = make_seed()
    with torch.no_grad():
        _x = _m(_x, steps=96)

    # Now: injure it, visibly.
    _before_damage = _x.clone()
    _dmg = damage_type.value
    if _dmg == 'hole (center)':
        _x[:, :, 14:26, 14:26] = 0
    elif _dmg == 'cut in half':
        _x[:, :, :, :GRID_SIZE // 2] = 0
    elif _dmg == 'scatter damage':
        _rng = np.random.default_rng(3)
        for _ in range(18):
            _cx, _cy = _rng.integers(6, GRID_SIZE - 6, size=2)
            _x[:, :, _cy - 3:_cy + 3, _cx - 3:_cx + 3] = 0
    elif _dmg == 'erase top':
        _x[:, :, :GRID_SIZE // 2, :] = 0

    # Heal: run the same network forward. Nothing tells it to heal.
    with torch.no_grad():
        _, _heal_traj = _m(_x, steps=80, return_trajectory=True)

    # Build the story-frames: mature → damage → healing
    _story = [_before_damage, _before_damage, _x, _x] + _heal_traj[::2]
    _src = frames_to_gif_b64(_story, scale=10, duration=80, loop_delay=1800)
    mo.Html(img_tag(_src,
        caption='mature → injury → recovery · the same network, no instruction'))
    return


@app.cell
def _(mo):
    mo.Html("""
    <div class="prose">
      <p>Nothing has told these cells what to do. There is no copy of the
      shape stored anywhere. The cells are, in a very real sense,
      <em>remembering the shape by recreating it</em>.</p>
      <p>The information is distributed. It's local. It's everywhere and
      nowhere.</p>
    </div>
    """)
    return


@app.cell
def _(mo):
    mo.Html("""
    <div class="prose">
      <div class="chapter">An original contribution</div>
      <div class="h1" style="font-size:2rem;">The sound of thinking</div>
      <p>The network has 16 channels per cell, but only the first 4 are
      visible to us — R, G, B, and alpha. The other 12 are where the <em>thinking</em>
      happens. They don't look like anything. Nobody sees them.</p>
      <p>I want you to see them.</p>
      <p>Below: the same healing process, but this time with channel 4 shown
      on the right. Watch what travels across the body just before the
      shape re-forms. You are looking at information propagating — the
      closest thing a cellular automaton has to a thought.</p>
    </div>
    """)
    return


@app.cell
def _(
    AVAILABLE_TARGETS,
    GRID_SIZE,
    MODELS,
    img_tag,
    make_seed,
    mo,
    render_side_by_side,
    torch,
):
    _target = 'lizard' if 'lizard' in AVAILABLE_TARGETS else AVAILABLE_TARGETS[0]
    _m = MODELS[_target]

    _x = make_seed()
    with torch.no_grad():
        _x = _m(_x, steps=96)
    # Cut in half — maximally dramatic
    _x[:, :, :, :GRID_SIZE // 2] = 0
    with torch.no_grad():
        _, _traj = _m(_x, steps=80, return_trajectory=True)

    _src = render_side_by_side(_traj[::2], _traj[::2], scale=8, duration=85)
    mo.Html(img_tag(_src, max_width=680,
        caption='left: what we see · right: what the organism is "thinking" '
                '(hidden channel 4)'))
    return


@app.cell
def _(mo):
    mo.Html("""
    <div class="prose">
      <p>Those ripples are real. They are the local messages — the cell-to-cell
      disagreements — that propagate outward from the injury site. Where the
      signal is strong, the cells are confused. They are negotiating what to
      become.</p>
      <p>Once the argument settles, the shape returns.</p>
    </div>
    """)
    return


@app.cell
def _(mo):
    mo.Html('<div class="prose"><div class="chapter">Chapter VI</div>'
            '<div class="h1" style="font-size:2rem;">Your turn</div>'
            '<p>All the knobs. All the organisms. Damage them however you want. '
            'Watch them disagree with you.</p></div>')
    return


@app.cell
def _(AVAILABLE_TARGETS, mo):
    play_target = mo.ui.dropdown(options=AVAILABLE_TARGETS,
                                 value=AVAILABLE_TARGETS[0],
                                 label='organism')
    play_damage = mo.ui.dropdown(
        options=['none', 'hole (center)', 'cut in half',
                 'scatter damage', 'erase top', 'erase bottom',
                 'diagonal cut'],
        value='none',
        label='injury')
    play_fire = mo.ui.slider(start=0.1, stop=1.0, step=0.1, value=0.5,
                             label='fire rate (synchrony)', show_value=True)
    play_steps = mo.ui.slider(start=32, stop=200, step=8, value=128,
                              label='steps', show_value=True)
    play_btn = mo.ui.run_button(label='run')
    mo.vstack([
        mo.hstack([play_target, play_damage], justify='start', gap=2),
        mo.hstack([play_fire, play_steps], justify='start', gap=2),
        play_btn,
    ], gap=1)
    return play_btn, play_damage, play_fire, play_steps, play_target


@app.cell
def _(
    GRID_SIZE,
    MODELS,
    frames_to_gif_b64,
    img_tag,
    make_seed,
    mo,
    np,
    play_btn,
    play_damage,
    play_fire,
    play_steps,
    play_target,
    torch,
):
    mo.stop(not play_btn.value, mo.md(''))

    _m = MODELS[play_target.value]
    _x = make_seed()

    # Grow to maturity first
    with torch.no_grad():
        _, _grow = _m(_x, steps=96, return_trajectory=True,
                      fire_rate=float(play_fire.value))
    _x = _grow[-1]

    # Apply damage
    _d = play_damage.value
    if _d == 'hole (center)':
        _x[:, :, 14:26, 14:26] = 0
    elif _d == 'cut in half':
        _x[:, :, :, :GRID_SIZE // 2] = 0
    elif _d == 'scatter damage':
        _rng = np.random.default_rng()
        for _ in range(20):
            _cx, _cy = _rng.integers(6, GRID_SIZE - 6, size=2)
            _x[:, :, _cy - 3:_cy + 3, _cx - 3:_cx + 3] = 0
    elif _d == 'erase top':
        _x[:, :, :GRID_SIZE // 2, :] = 0
    elif _d == 'erase bottom':
        _x[:, :, GRID_SIZE // 2:, :] = 0
    elif _d == 'diagonal cut':
        _mask = np.triu(np.ones((GRID_SIZE, GRID_SIZE)), k=0).astype(bool)
        _x[:, :, _mask] = 0

    with torch.no_grad():
        _, _heal = _m(_x, steps=int(play_steps.value), return_trajectory=True,
                      fire_rate=float(play_fire.value))

    # Story: growth → damage → heal
    _frames = _grow[::4] + [_x] * 3 + _heal[::max(1, len(_heal) // 40)]
    _src = frames_to_gif_b64(_frames, scale=10, duration=65, loop_delay=1200)
    mo.Html(img_tag(_src, caption='grow · injure · observe'))
    return


@app.cell
def _(mo):
    mo.Html("""
    <div class="prose" style="margin-top: 4rem;">
      <div class="chapter">Try it yourself</div>
      <div class="h1" style="font-size:2rem;">Draw the wound</div>
      <p>Pick an organism below. It will grow to maturity. Then
      <em>paint directly on it</em> with your mouse to erase cells — as many or
      as few as you like. Hit <strong>heal</strong> and watch it decide what to do.</p>
    </div>
    """)
    return


@app.cell
def _():
    import anywidget
    import traitlets as _tr

    class _PW(anywidget.AnyWidget):
        _esm = """
        function render({ model, el }) {
            const SCALE = 10, G = 32;
            const canvas = document.createElement('canvas');
            canvas.width = G * SCALE; canvas.height = G * SCALE;
            canvas.style.cssText = 'cursor:crosshair;image-rendering:pixelated;border:1px solid #2a3a5a;border-radius:4px;display:block;';
            const ctx = canvas.getContext('2d');
            const painted = new Set();

            function redraw() {
                const src = model.get('img_src');
                if (!src) return;
                const img = new Image();
                img.onload = () => {
                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                    ctx.fillStyle = 'rgba(0,0,0,1)';
                    for (const [gx, gy] of model.get('mask'))
                        ctx.fillRect(gx*SCALE, gy*SCALE, SCALE, SCALE);
                };
                img.src = src;
            }

            function getCell(e) {
                const r = canvas.getBoundingClientRect();
                return [
                    Math.floor((e.clientX - r.left) * canvas.width / r.width / SCALE),
                    Math.floor((e.clientY - r.top) * canvas.height / r.height / SCALE)
                ];
            }

            function paint(e) {
                const [gx, gy] = getCell(e);
                if (gx < 0 || gx >= G || gy < 0 || gy >= G) return;
                // Paint a 2×2 brush for easier drawing
                for (let dx = 0; dx < 2; dx++) for (let dy = 0; dy < 2; dy++) {
                    const bx = gx + dx, by = gy + dy;
                    if (bx >= G || by >= G) continue;
                    const k = `${bx},${by}`;
                    if (painted.has(k)) continue;
                    painted.add(k);
                    ctx.fillStyle = 'rgba(0,0,0,1)';
                    ctx.fillRect(bx*SCALE, by*SCALE, SCALE, SCALE);
                }
                const newCells = [...painted].map(k => k.split(',').map(Number));
                model.set('mask', newCells);
                model.save_changes();
            }

            let down = false;
            canvas.addEventListener('mousedown', e => { down = true; paint(e); });
            canvas.addEventListener('mousemove', e => { if (down) paint(e); });
            canvas.addEventListener('mouseup', () => down = false);
            canvas.addEventListener('mouseleave', () => down = false);
            canvas.addEventListener('touchstart', e => { e.preventDefault(); down = true; paint(e.touches[0]); }, {passive:false});
            canvas.addEventListener('touchmove', e => { e.preventDefault(); if (down) paint(e.touches[0]); }, {passive:false});
            canvas.addEventListener('touchend', () => down = false);

            const hint = document.createElement('p');
            hint.textContent = 'paint to damage  →  then click heal';
            hint.style.cssText = 'font-size:0.8rem;color:#8fa3bd;font-style:italic;margin:0 0 8px 0;font-family:Inter,system-ui,sans-serif;';

            const clearBtn = document.createElement('button');
            clearBtn.textContent = 'clear damage';
            clearBtn.style.cssText = 'margin-top:8px;padding:4px 12px;font-size:0.75rem;cursor:pointer;background:#0d1117;color:#8fa3bd;border:1px solid #2a3a5a;border-radius:4px;';
            clearBtn.onclick = () => { painted.clear(); model.set('mask', []); model.save_changes(); redraw(); };

            el.appendChild(hint);
            el.appendChild(canvas);
            el.appendChild(clearBtn);
            model.on('change:img_src', () => { painted.clear(); redraw(); });
            redraw();
        }
        export default { render };
        """
        img_src = _tr.Unicode("").tag(sync=True)
        mask    = _tr.List([]).tag(sync=True)

    PaintWidget = _PW
    return (PaintWidget,)


@app.cell
def _(AVAILABLE_TARGETS, mo):
    draw_target = mo.ui.dropdown(
        options=AVAILABLE_TARGETS,
        value='heart' if 'heart' in AVAILABLE_TARGETS else AVAILABLE_TARGETS[0],
        label='organism',
    )
    draw_heal_btn = mo.ui.run_button(label='heal')
    return draw_heal_btn, draw_target


@app.cell
def _(
    Image,
    MODELS,
    PaintWidget,
    base64,
    draw_heal_btn,
    draw_target,
    io,
    make_seed,
    mo,
    to_rgba_img,
    torch,
):
    # Grow to maturity so the canvas always shows a fully-grown organism
    _seed = make_seed()
    with torch.no_grad():
        _grown = MODELS[draw_target.value](_seed, steps=96)

    # Render to PNG base64 for the canvas widget
    _arr = to_rgba_img(_grown[0])
    _img = Image.fromarray(_arr, 'RGBA')
    _img = _img.resize((_arr.shape[1] * 10, _arr.shape[0] * 10), Image.NEAREST)
    _bg = Image.new('RGBA', _img.size, (10, 12, 18, 255))
    _bg.alpha_composite(_img)
    _buf = io.BytesIO()
    _bg.convert('RGB').save(_buf, format='PNG')
    _img_b64 = 'data:image/png;base64,' + base64.b64encode(_buf.getvalue()).decode()

    draw_widget = PaintWidget(img_src=_img_b64)
    draw_grown_np = _grown.detach().numpy()

    mo.vstack([
        mo.hstack([draw_target, draw_heal_btn], justify='start', gap=2),
        draw_widget,
    ])
    return draw_grown_np, draw_widget


@app.cell
def _(
    GRID_SIZE,
    MODELS,
    draw_grown_np,
    draw_heal_btn,
    draw_target,
    draw_widget,
    frames_to_gif_b64,
    img_tag,
    mo,
    torch,
):
    mo.stop(not draw_heal_btn.value, mo.md(''))

    _m = MODELS[draw_target.value]
    _x = torch.from_numpy(draw_grown_np.copy())

    # Apply drawn mask — zero out every painted cell
    for _gx, _gy in draw_widget.mask:
        if 0 <= _gx < GRID_SIZE and 0 <= _gy < GRID_SIZE:
            _x[:, :, _gy, _gx] = 0.0

    # If nothing was drawn, fall back to a center hole so the button still does something
    if not draw_widget.mask:
        _x[:, :, 12:20, 12:20] = 0.0

    _before = torch.from_numpy(draw_grown_np.copy())
    with torch.no_grad():
        _, _traj = _m(_x, steps=96, return_trajectory=True)

    _frames = [_before, _before, _x, _x] + _traj[::2]
    _src = frames_to_gif_b64(_frames, scale=10, duration=80, loop_delay=2000)
    mo.Html(img_tag(_src, caption='your damage · their recovery'))
    return


@app.cell
def _(mo):
    mo.Html("""
    <div class="prose" style="margin-top: 5rem;">
      <div class="chapter">Closing</div>
      <div class="h1" style="font-size:2rem;">What was this?</div>

      <p>Nothing in this notebook is alive. There are no cells, no membranes,
      no DNA. Just numbers — sixteen floating-point values per pixel, updated
      by a tiny neural network — simulating something that looks
      <em>awfully</em> like life.</p>

      <p>And yet: a seed grows. An injury heals. Local rules, repeated
      everywhere, produce global order. No architect. No blueprint. No one in
      charge.</p>

      <p>That is not a simulation of biology. That <em>is</em> biology, or a
      big enough piece of it that it starts to feel a little uncanny to
      watch.</p>

      <p>Maybe 'life' is not a substance we need to contain. Maybe it is just
      a pattern that emerges whenever the rules are right and the neighbors
      keep talking.</p>

      <p style="text-align:center; color:#8aa4c8; margin-top: 3rem;">
        Thanks for watching them talk.
      </p>
    </div>

    <div class="prose" style="margin-top: 3rem; font-size:0.85rem;
                              color:#7891a8; text-align:center;">
      <p>Based on "Growing Neural Cellular Automata" (Mordvintsev, Randazzo,
      Niklasson & Levin, 2020). Extension — latent channel visualization during
      healing — is original to this notebook. Built with marimo and PyTorch.
      All models trained from scratch on Nebius GPU.</p>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""

    """)
    return


if __name__ == "__main__":
    app.run()
