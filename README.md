# Unsupervised Music Clustering (Audio + Lyrics)

This repo builds a multilingual music dataset (English + Bangla included) and runs an unsupervised clustering pipeline using VAE-based embeddings and baselines.

Key capabilities:

- **Features**: audio (MFCC summary, MFCC frames, log-mel spectrogram), lyrics (TF-IDF), hybrid audio+lyrics, and a simple multimodal variant.
- **Representation learning**: VAE / Beta-VAE (`--beta`) / Conditional VAE (`--cond-col`) with MLP or Conv2D backbones.
- **Baselines**: PCA (`--baseline pca`), Autoencoder (`--baseline ae`), Raw features (`--baseline raw`).
- **Clustering**: KMeans, Agglomerative, DBSCAN, Spectral.
- **Evaluation**: silhouette / Calinski-Harabasz / Davies-Bouldin (internal), plus ARI/NMI/purity when you provide a ground-truth label column.

## How the pipeline works (high-level)

The main entrypoint is `python -m src.run_pipeline ...`.

1. Load a metadata CSV (e.g. `data/metadata_*.csv`).
2. Build features from audio and/or lyrics.
3. Optionally learn an embedding (`VAE`, `AE`, or `PCA`).
4. Cluster the embeddings.
5. Write metrics, latents, and visualizations into `--outdir`.

Important: this project is **unsupervised** for clustering, but can still compute supervised cluster metrics if a label column exists (e.g. `--label-col language`).

## Features (what `--feature` means)

All supported feature modes are implemented in `src/run_pipeline.py` and `src/features.py`.

| `--feature`    | Data used                                    |                      Output representation | Typical VAE architecture | Notes                                                                                                                   |
| -------------- | -------------------------------------------- | -----------------------------------------: | ------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| `lyrics_tfidf` | `lyrics_clean` (fallback `lyrics`)           | 2D dense TF‑IDF matrix `(N, max_features)` | `mlp`                    | Defaults to character n‑grams (`char_wb`, 3–5) to work well for mixed scripts (English + Bangla).                       |
| `mfcc`         | `audio_path`                                 |     2D MFCC summary vector `(N, n_mfcc*4)` | `mlp`                    | Each track becomes a fixed vector by concatenating per‑coefficient mean/std/min/max.                                    |
| `mfcc_frames`  | `audio_path`                                 |      4D tensor `(N, 1, n_mfcc, tf_frames)` | `conv2d`                 | MFCC time series padded/truncated to `--tf-frames`.                                                                     |
| `logmelspec`   | `audio_path`                                 |      4D tensor `(N, 1, n_mels, tf_frames)` | `conv2d`                 | Log‑mel spectrogram padded/truncated to `--tf-frames`.                                                                  |
| `hybrid`       | `audio_path` + lyrics                        |         2D concat embedding `(N, Da + Dl)` | audio: `mlp`/`conv2d`    | Audio is embedded by a VAE (choice via `--audio-feature`). Lyrics are embedded via TF‑IDF → SVD (`--lyrics-embed-dim`). |
| `multimodal`   | `audio_path` + lyrics + categorical metadata |    2D concat embedding `(N, Da + Dl + Dg)` | audio: `mlp`/`conv2d`    | Like `hybrid`, plus a one‑hot encoding from `--genre-col` (if you choose to use it). Missing values become `unknown`.   |

## Models / baselines (what `--baseline`, `--vae-arch` mean)

### Representation learners

| Mode                   | How to enable                    | Architecture                                                 | What it learns/does                                                                                       |
| ---------------------- | -------------------------------- | ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------- |
| VAE                    | `--baseline none`                | `--vae-arch mlp` (vector) or `--vae-arch conv2d` (time‑freq) | Learns a latent embedding `z` by reconstruction + KL regularization.                                      |
| Beta‑VAE               | `--baseline none --beta <value>` | same as VAE                                                  | Same as VAE but scales KL term by `beta` to encourage more factorized/disentangled latents.               |
| Conditional VAE (CVAE) | `--cond-col <col>`               | same as VAE                                                  | Conditions the encoder/decoder on a one‑hot vector from a metadata column (e.g. `language`).              |
| Autoencoder (AE)       | `--baseline ae`                  | same backbone as VAE                                         | Same network structure, but trained deterministically (no sampling) and without KL (reconstruction only). |
| PCA baseline           | `--baseline pca`                 | (sklearn)                                                    | Flattens tensors when needed and projects to up to 32 dims (bounded by `N` and input dim).                |
| Raw baseline           | `--baseline raw`                 | (none)                                                       | Uses standardized features directly as the clustering embedding (tensors are flattened).                  |

### Backbones

| Backbone   | `--vae-arch` | Used for                                                 | What it is                                                                      |
| ---------- | ------------ | -------------------------------------------------------- | ------------------------------------------------------------------------------- |
| MLP VAE    | `mlp`        | 2D features (`lyrics_tfidf`, `mfcc`, and vector modes)   | Fully connected encoder/decoder with hidden sizes from `VAEConfig.hidden_dims`. |
| Conv2D VAE | `conv2d`     | 4D time‑frequency features (`mfcc_frames`, `logmelspec`) | Strided 2D conv encoder + transposed‑conv decoder over `(freq, time)` maps.     |

## Clustering methods

| `--cluster-method` | Notes                                                                                                                 |
| ------------------ | --------------------------------------------------------------------------------------------------------------------- |
| `kmeans`           | Fast default, requires `--clusters`.                                                                                  |
| `agglomerative`    | Hierarchical clustering, requires `--clusters`.                                                                       |
| `dbscan`           | Density clustering; may label noise as `-1` (shown as “noise” in plots). Uses `--dbscan-eps`, `--dbscan-min-samples`. |
| `spectral`         | Graph-based clustering; uses nearest-neighbor affinity with `--spectral-n-neighbors`.                                 |

## Data manifests (already in `data/`)

The pipeline is driven by CSV manifests. The most important ones:

- `data/metadata_audio_lyrics_mixed.csv`: paired audio+lyrics (multilingual).
- `data/metadata_known_categories_preprocessed_balanced.csv`: lyrics-only dataset used for the Easy task (balanced; good for en vs bn clustering).
- `data/metadata.csv`: default manifest written by the downloader.

Expected columns (depending on feature mode):

- Required: `id`
- Audio features: `audio_path`
- Lyrics features: `lyrics` (or `lyrics_clean` if present)
- Labels for evaluation: `label` (or pass `--label-col language`, etc.)
- Optional metadata for plots: `language` (and any other column you choose to plot via `--dist-cols`)

## Setup (Windows / PowerShell)

```powershell
cd E:\JetBrains\project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Download JamendoLyrics subset (optional)

Downloads audio into `data/audio/`, lyrics into `data/lyrics/`, and writes `data/metadata.csv`:

```powershell
python download.py --language English
```

You can keep all languages:

```powershell
python download.py --language all
```

If Hugging Face access is gated:

```powershell
huggingface-cli login
```

## Run all experiment suites (Easy / Medium / Hard)

Canonical runner:

```powershell
python scripts\run_all_tasks.py --device cpu
```

Compatibility shortcut (forwards to the same runner):

```powershell
python run_all_tasks.py --device cpu
```

Quick minimal run (no visualizations):

```powershell
python scripts\run_all_tasks.py --device cpu --viz none
```

Where results go (created next to the runner script):

- `scripts/results_easy_task/`
- `scripts/results_medium_task/`
- `scripts/results_hard_task/`

Each task folder contains per-run subfolders plus an aggregated comparison CSV:

- `easy_task_comparison.csv`
- `medium_task_comparison.csv`
- `hard_task_comparison.csv`

## Run a single pipeline experiment

Lyrics-only TF-IDF example:

```powershell
python -m src.run_pipeline ^
  --metadata data\metadata_known_categories_preprocessed_balanced.csv ^
  --feature lyrics_tfidf ^
  --baseline pca ^
  --clusters 2 ^
  --label-col language ^
  --viz none ^
  --outdir scripts\scratch_run
```

Audio log-mel ConvVAE example:

```powershell
python -m src.run_pipeline ^
  --metadata data\metadata_audio_lyrics_mixed.csv ^
  --feature logmelspec ^
  --vae-arch conv2d ^
  --baseline none ^
  --clusters 6 ^
  --label-col language ^
  --viz tsne ^
  --outdir scripts\scratch_audio
```

## Utility scripts

- `scripts/preprocess_lyrics_mixed.py`: writes a `lyrics_clean` column for cleaner TF-IDF.
- `scripts/compare_baseline.py`: compares the latest `vae` vs `pca` rows inside a metrics CSV (pass `--metrics` if your output folder is not `results/`).
- `scripts/tune_lyrics_vae.py`: optional tuning helper for lyrics VAE.
- `scripts/fetch_bangla_lyrics.py`: optional Bangla lyrics scraper template.
  - Requires extra deps: `pip install requests beautifulsoup4` (and optionally `langid`).

## Outputs

Every pipeline run writes:

- `clustering_metrics.csv` (appended per run)
- `latents.npz` (embeddings + ids)
- `latent_visualization/` (plots when `--viz tsne|umap`)
- `reconstructions/` (when `--save-recon`)

### What files mean (including every plot filename)

All output paths below are relative to `--outdir`.

| File / pattern                                          | Produced when                                | Meaning                                                                                                                                                         |
| ------------------------------------------------------- | -------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `clustering_metrics.csv`                                | always                                       | One row per run with run settings + clustering metrics.                                                                                                         |
| `latents.npz`                                           | always                                       | Numpy bundle with `z` (embeddings), `ids`, `clusters`, plus `mean/std` used for standardization (when applicable).                                              |
| `latent_visualization/{run}_{cluster_method}_{viz}.png` | `--viz tsne` or `--viz umap`                 | 2D projection of the embedding space. Points are colored by predicted cluster. Use this to visually inspect cluster separation and outliers.                    |
| `cluster_distributions/clusters_by_{col}.png`           | when `--dist-cols` contains existing columns | Stacked bar chart: for each value of `{col}`, how many items fell into each predicted cluster. Useful for checking whether clusters correlate with that column. |
| `cluster_distributions/clusters_by_{col}.png`           | when `--dist-cols` contains existing columns | Stacked bar chart: for each value of `{col}`, how many items fell into each predicted cluster. Useful for checking whether clusters correlate with that column. |
| `reconstructions/recon_examples.png`                    | `--save-recon` (VAE/AE runs)                 | Reconstruction examples: original vs reconstructed feature vectors (line plots) or time‑freq maps (heatmaps). If recon is poor, latents may be less meaningful. |

Notes on `{run}` in filenames:

- `{run}` is `vae` when `--baseline none`, otherwise it matches `--baseline` (e.g. `pca`, `ae`, `raw`).
- `{viz}` is `tsne` or `umap`.
- For DBSCAN, the “noise” cluster is labeled as `-1`.

Easy task runner note:

- `scripts/run_all_tasks.py` also generates **both** `latent_visualization/clusters_tsne.png` and `latent_visualization/clusters_umap.png` for each Easy-task run (unless you pass `--viz none`).

Those two images mean the same thing as the `{run}_{cluster_method}_{viz}.png` plot, but are generated as a post-processing step from the saved `latents.npz` to guarantee both projections exist.

## Report

The project writeup is in `PROJECT_REPORT.md`.
