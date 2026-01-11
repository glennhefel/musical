# Unsupervised Learning Project: VAE for Hybrid Language Music Clustering

This project implements unsupervised clustering of music using Variational Autoencoders (VAEs) and classical baselines, supporting lyrics-only and hybrid (audio+lyrics) experiments. Audio files are omitted for repository sharing.

## How to Reproduce Results

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

2. **Run all experiments and generate results:**
   ```sh
   python run_all_tasks.py
   ```
   This will:
   - Extract features (from available lyrics and metadata)
   - Train VAE and baseline models
   - Perform clustering (K-Means, Agglomerative, DBSCAN, Spectral)
   - Evaluate and save metrics/visualizations

3. **View results:**
   - Metrics, figures are saved in the `results/` directory and catagorized as easy medium and hard tasks according to the specifications 
   - Other saved files: clustering_metrics.csv (appended per run)
     latents.npz (embeddings + ids)
     latent_visualization/ (plots when --viz tsne|umap)
     reconstructions/ (when --save-recon)

4. **Features:**
Audio (MFCC summary, MFCC frames, log-mel spectrogram), lyrics (TF-IDF), hybrid audio+lyrics, and a simple multimodal variant.
Representation learning: VAE / Beta-VAE (--beta) / Conditional VAE (--cond-col) with MLP or Conv2D backbones.
Baselines: PCA (--baseline pca), Autoencoder (--baseline ae), Raw features (--baseline raw).
Clustering: KMeans, Agglomerative, DBSCAN, Spectral.
Evaluation: silhouette / Calinski-Harabasz / Davies-Bouldin (internal), plus ARI/NMI/purity when you provide a ground-truth label column.

## Notes
- Audio files are not included in this repository. Download them here and put them in the data folde.
- For full audio+lyrics experiments, you must supply your own audio files in the expected folder structure (see `data/` and `src/dataset.py`).
- All code is compatible with Python 3.8+.

## Citation
If you use this code or pipeline, please cite or reference the original repository and paper.
