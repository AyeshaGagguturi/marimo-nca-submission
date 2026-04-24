# Storyboard — "The Language of Life"

> *A marimo notebook that wins because it makes people feel something before
> it makes them learn something.*

## Core premise

Most NCA demos are technical. They show growth, they show healing, they
stop there. This notebook treats Neural Cellular Automata as a *parable* —
the simplest possible version of how a body builds itself — and uses that
parable to teach the research. The technical content is all present, but it
arrives wrapped in a story the judges actually want to keep reading.

## The emotional arc

Seven chapters, each doing one job:

| Ch. | Title                                  | Purpose                              | What the judge feels                 |
|-----|----------------------------------------|--------------------------------------|--------------------------------------|
| —   | *Opening*                              | set the stakes                       | "oh — this is about *us*"            |
| I   | A single cell                          | establish the seed                   | curiosity                            |
| II  | Rules without rulers                   | Conway → show the concept is old     | delight, nostalgia                   |
| III | What if the rules could learn?         | untrained → trained reveal           | the first "wow"                      |
| IV  | Becoming                               | interactive growth, multiple targets | ownership, play                      |
| V   | It remembers what it's supposed to be  | **healing — the peak**               | awe                                  |
| V+  | The sound of thinking (**original**)   | latent channels during healing       | "I've never seen this before"        |
| VI  | Your turn                              | full playground                      | engagement, experimentation          |
| —   | Closing                                | tie the metaphor together            | stays with them after closing tab    |

The two load-bearing moments are **the untrained → trained cut** in Ch. III
(the reveal) and **the healing sequence** in Ch. V (the emotional peak).
Everything else frames those two moments.

## The original contribution — "thought waves"

Every NCA demo shows the RGB channels. Nobody shows the hidden channels
during healing. Channel 4 (arbitrary — any of channels 4–15 works) reveals
information waves propagating outward from the damage site as the cells
negotiate what to become. Rendered as a heatmap beside the RGB view, it
makes the invisible visible.

This is a genuine research-adjacent contribution: the paper notes that
information must propagate somehow, but doesn't visualize it. We do.
Judging criteria explicitly reward "custom extensions that provide insight
into the topic" — this is exactly that.

## Visual design rules

- **Dark background throughout** (10,12,18). Makes the organisms feel alive
  against void. Every frame is premultiplied alpha composited on this.
- **Pixel-art scale**: 10× nearest-neighbor upscaling. Preserves the grid
  texture. Never blur.
- **Georgia serif** for prose, **Inter** for labels, **one accent color**
  (`#ffd78a` — warm gold) for emphasized words. No other colors in text.
- **Captions** italicized, below every image, small and quiet.
- **Breathing room**: wide margins, 680px max prose width. Don't cram.
- **One image at a time**. No side-by-side grids of comparisons except for
  the thought-wave visualization (where it's earned).

## Technical choices and why

| Choice                           | Why                                                       |
|----------------------------------|-----------------------------------------------------------|
| Pre-trained weights loaded       | Training in-notebook kills the pacing. Train on Nebius.   |
| 40×40 grid, 16 channels          | Canonical from the paper. Fast inference on CPU.          |
| GIF with PIL (not matplotlib)    | Plays inline, loops cleanly, no backend flakiness.        |
| Pool training + random damage    | Makes healing robust enough to demo.                      |
| Zero-init final layer            | Without this, training diverges on step 1. Every time.    |
| Per-param grad normalization     | Non-obvious stability trick from the paper. Essential.    |
| `mo.ui.run_button` (not auto)    | Expensive cells only run on click. Notebook stays snappy. |

## Shippable-in-48-hours budget

### Day 1 — training and core render (8–10 hrs)

1. **Hour 0–2**: `train.py` on Nebius. Start all 5 targets training in parallel
   (5 processes, or 1 H100 sequential takes ~40 min total). Verify losses
   converging.
2. **Hour 2–5**: Set up marimo locally, wire up the NCA class, verify loaded
   weights produce correct growth. Get one GIF rendering in a cell.
3. **Hour 5–8**: Build chapters I–IV. This is the "setup" half.
4. **Hour 8–10**: Validate healing works cleanly on all 5 targets. If scatter
   damage fails on any, retrain with more aggressive damage augmentation.

### Day 2 — the emotional half (8–10 hrs)

1. **Hour 10–13**: Build chapter V. The healing demo is the *hardest part
   visually* — it must look effortless and dramatic. Tweak frame timing,
   `loop_delay`, damage severity. Test with fresh eyes.
2. **Hour 13–15**: Side-by-side thought-wave render. This is your
   differentiator — spend the time.
3. **Hour 15–17**: Playground cell. Limit damage types to ones that heal
   reliably — better to show 4 that work perfectly than 8 that flicker.
4. **Hour 17–19**: Prose pass. Read every piece of text aloud. Cut anything
   that sounds like an AI wrote it. Shorten.
5. **Hour 19–20**: Upload to molab, test end-to-end in the browser (WASM!).
   Check that all GIFs render, no broken cells.

### Hour 21+ — polish

- Add a favicon / social preview image (first frame of lizard growth).
- Double-check all captions.
- Submit.

## What to cut if time runs out

Drop in this order:

1. **Fire rate slider in playground** — interesting but not essential.
2. **Untrained NCA in Ch. III** — emotionally useful but not load-bearing.
3. **Conway GoL in Ch. II** — the nicest thing to drop last, because it
   grounds the story in something judges recognize.

Do **not** drop:
- The healing sequence (Ch. V)
- The thought-wave visualization (the contribution)
- The closing prose

## What a winning reading looks like

A judge opens the notebook at 11pm the night before results.
- They read the opening and think "okay, this is something."
- They hit the first GIF in Ch. III and lean in.
- By the end of Ch. V they're quiet.
- They play in Ch. VI for two minutes.
- They close the tab and remember it the next morning.

That is the target. Optimize every decision for that sequence.

## Submission checklist

- [ ] All 5 weights trained and uploaded to molab
- [ ] Notebook runs end-to-end in the molab WASM environment
- [ ] All GIFs render under 2MB each (optimize palette, frame stride)
- [ ] Prose read aloud and cut
- [ ] First frame of opening shows the "single cell" image — this is what
      previews on Twitter when shared
- [ ] Extension (thought waves) is clearly labeled as an original contribution
- [ ] Paper citation present at the bottom
- [ ] molab link is public and tested in incognito

## References

- Mordvintsev, Randazzo, Niklasson, Levin. *Growing Neural Cellular
  Automata*. Distill, 2020. https://distill.pub/2020/growing-ca/
- Randazzo, Mordvintsev, Niklasson, Levin, Fouts. *Self-classifying MNIST
  Digits*. Distill, 2020. https://distill.pub/2020/selforg/mnist/
  (relevant for the multi-target extension, if time permits)
- alphaXiv × marimo competition: https://marimo.io/pages/events/notebook-competition
