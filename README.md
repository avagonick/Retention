# Neuroscience-Guided Educational Video Optimizer

A self-improving loop that optimizes educational videos for engagement, using a
neuroscience foundation model (**TRIBE v2**) as the fitness function.

> *Brainrot is destroying attention spans — so we use a brain-encoding model to
> scientifically optimize educational content for how the brain actually responds.*

## How it works

1. Take an educational video (`.mp4`).
2. **Score** it with TRIBE v2 → predicted fMRI cortical brain activity over time.
3. Reduce that to a per-network **engagement** curve (the fitness function).
4. *(loop, in progress)* An LLM reads the score → proposes edits; Pika generates
   inserted clips; Deepgram generates narration; ffmpeg stitches; re-score; repeat
   until engagement stops improving.

## Architecture: what runs where

- **TRIBE v2** runs *remotely* on a Lightning AI GPU (A100), exposed as an HTTP
  API. It does all the heavy brain-encoding compute. We never run it locally.
- **Everything else runs locally** and calls TRIBE over HTTP: visualization,
  network reduction, the engagement metric, and (soon) the LLM/Pika/Deepgram/ffmpeg
  edit loop.

TRIBE is Meta's tri-modal brain encoder (video + audio + text). It returns a
`(T, 20484)` float32 array: `T` ≈ one row per second of video, `20484` cortical
vertices (fsaverage5; cols 0–10241 = left hemisphere, 10242–20483 = right).
Values are z-scored predicted BOLD (signed, centred near 0). License: CC-BY-NC.

## Repository layout

| File | Role |
|------|------|
| `config.py` | Loads `.env`; exposes `BASE_URL` / `TOKEN`. Secrets stay out of source. |
| `tribe_client.py` | **Remote boundary** — `score(video)` / `check_health()` over HTTP. |
| `brain_viz.py` | Renders the array on a cortical surface (PNG / interactive HTML / MP4). |
| `brain_networks.py` | Collapses 20,484 vertices → 5 named functional networks (Destrieux atlas). |
| `engagement.py` | The **fitness function**: attention + sensory − default-mode, lag-corrected. |
| `video_utils.py` | ffmpeg helpers (e.g. tail-padding to recover a clip's final seconds). |
| `main.py` | CLI: health → score → viz → networks → engagement, into `out/<video>/`. |

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # macOS/Linux

cp .env.example .env      # then edit .env with your TRIBE_API_TOKEN (and URL)
```

CPU only — no GPU/torch needed locally. First nilearn run auto-downloads the
fsaverage5 surface (~few MB, once).

## Usage

```bash
# full run: score remotely, then write all artifacts to out/<video-name>/
python main.py path/to/clip.mp4

# re-render viz/metrics from the cached array (no re-scoring):
python main.py path/to/clip.mp4 --reuse
```

Each run produces a self-contained folder:

```
out/<video-name>/
├── preds.npy              raw (T, 20484) prediction array
├── times.json            row start-times (sec)
├── brain_static.png      2×2 cortical snapshot (most-active timestep)
├── brain_interactive.html rotatable 3D whole brain
├── brain_activity.mp4    per-second animation (1 fps = real-time)
├── networks_timeseries.png  per-network response (line + heatmap)
├── networks.json         per-network time series
├── engagement.png        engagement fitness curve + components
├── engagement.json       LLM-ready payload (fitness, edit target, series)
└── manifest.json         run summary
```

## Notes

- Test clips should be **≥15–30s with real audio** for reliable predictions.
- TRIBE predictions trail the stimulus by ~5s (hemodynamic lag); the engagement
  layer corrects for this when localizing edits. To fully cover a clip's final
  seconds, tail-pad it with `video_utils.pad_video_tail` before scoring.
- The first `/score` after a cold server start takes ~2.5 min (lazy model load).

## Status

✅ Score · visualize · network reduction · engagement fitness
🚧 LLM edit-proposer → Pika generation → Deepgram narration → ffmpeg stitch → re-score loop
