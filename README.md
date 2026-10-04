# Recommender Systems

This repository contains two graduate-level projects on recommender systems: an optimized collaborative filtering study on MovieLens 100K, and a content-based system built from raw movie reviews at scale. Both were developed for CMP5104 (Recommender Systems) and focus on benchmarking classical architectures against neural and NLP-based alternatives.

---

## Folder Structure
```
recommender-systems/
├── ml-100k-parameter-optimizer/
│ ├── ml-100k-parameter-optimizer.ipynb
│ ├── learning-curve2.png
│ ├── embedding_analysis.png
│ ├── learning-curve2.png
│ └── report-assignment2-ml100kParamOpt.pdf
├── genome-2021-movie-recommender/
│ ├── content-based-movie-recommender.ipynb
│ └── report-assignment1-CBMR.pdf
└── README.md

```

---

## `ml-100k-parameter-optimizer`: Optimized Collaborative Filtering & NCF
**Dataset:** [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) - Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets. *ACM TIIS*.

### Methodology

- Implemented **five model architectures** in PyTorch: Bias-Only Baseline, Matrix Factorization (MF), MF with Bias, Neural Collaborative Filtering (NCF), and NCF without Bias.
- Entire dataset was converted to GPU-resident tensors to eliminate data transfer overhead during training.
- Conducted a **Grid Search over 125 hyperparameter combinations**: embedding dims `[8, 16, 32, 64, 128]`, learning rates `[0.001, 0.005, 0.01, 0.05, 0.1]`, and regularization strengths `[0.0, 1e-6, 1e-5, 1e-4, 1e-3]`.
- Optimization stack: **AdamW** (decoupled weight decay), **ReduceLROnPlateau** scheduler (halves LR after 2 stagnant epochs), and **Early Stopping** (patience = 3).
- Final benchmarking used **5-Fold Cross-Validation** on the optimal config: `Embedding Dim=32, LR=0.005, Reg=1e-6`.

### Results
Models were evaluated using 5-Fold Cross-Validation. The optimal configuration was found to be `Embedding Dim=32`, `LR=0.005`, and `Reg=1e-06`.

| Model Architecture | Test RMSE |
| :--- | :---: |
| **Matrix Factorization (w/ Bias)** | **0.9602** |
| Matrix Factorization (No Bias) | 0.9640 |
| Neural Collaborative Filtering | 1.7150 |
| NCF (No Bias) | 1.6069 |
| Bias Only Baseline | 2.7879 |

> **Key Finding:** The Matrix Factorization with Bias achieved the lowest RMSE and fastest convergence. NCF architectures plateaued significantly higher (~1.6), likely because data sparsity makes dot-product approaches more suitable than deep networks at this dataset size.

<p align="center">
  <img src="ml-100k-parameter-optimizer/learning-curve2.png" width="480"/>
  <br><em>Figure 1: Validation RMSE over 100 epochs. MF with Bias (green) converges fastest and achieves the lowest error.</em>
</p>

<p align="center">
  <img src="ml-100k-parameter-optimizer/embedding_analysis.png" width="480"/>
  <br><em>Figure 2: Embedding size vs. RMSE. Performance degrades beyond dim=32, indicating overfitting at higher capacities.</em>
</p>

---

## `genome-2021-movie-recommender`: Content-Based Recommender System (NLP)
**Dataset:** [MovieLens Tag Genome 2021](https://grouplens.org/datasets/movielens/tag-genome-2021/) — Vig, J., Sen, S., & Riedl, J. (2012). The Tag Genome. ACM TIIS.

### Methodology

- Processed **2.6 million raw movie reviews** across 52,081 movies; text was cleaned (punctuation, numbers, stop words removed) and aggregated per movie.
- Processed corpus was checkpointed to a **Parquet file** to avoid repeated preprocessing on Colab restarts.
- Pre-computing a full 52K×52K similarity matrix caused memory crashes; resolved with an **on-the-fly similarity calculation** that only evaluates pairs relevant to each prediction.
- Implemented two text representation strategies:
  1. **TF-IDF + Cosine Similarity** — weights terms by importance relative to the corpus.
  2. **Binary Counts + Jaccard Similarity** — weights terms by simple presence/absence overlap.
- Top-N evaluation used a **user profile vector** (mean of liked-movie content vectors) compared against the full catalog to generate Top-10 lists.

###Results
1.  **TF-IDF + Cosine Similarity:** Weighs words by importance/frequency.
2.  **Binary Counts + Jaccard Similarity:** Weighs words by simple presence/overlap.

### Results

Evaluated on a 5,000-rating sample from a 20% held-out test set.

| Model Strategy | MAE | RMSE | Hit Ratio (@10) |
| :--- | :---: | :---: | :---: |
| **Binary + Jaccard** | **0.7593** | **0.9850** | N/A |
| TF-IDF + Cosine | 0.7607 | 0.9855 | 0.20% |
| Global Average Baseline | 0.8355 | 1.0545 | - |
| Random Baseline | 1.5861 | 1.9565 | - |

> **Key Finding:** Both content-based models outperformed all baselines. Binary + Jaccard marginally edged out TF-IDF, suggesting word presence is a stronger signal than frequency weighting for this review corpus. The 0.2% Hit Ratio for TF-IDF, while low in absolute terms, is realistic for pure content-based retrieval over a 52K+ item catalog.

---

## Reproduction

### Requirements

```bash
pip install torch scikit-learn pandas numpy nltk matplotlib seaborn
```

> A **Google Colab Pro** environment (High-RAM + GPU runtime) is recommended for both projects due to dataset size and grid search compute requirements.

### `ml-100k-parameter-optimizer`

1. Download [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) and place the `u.data` splits under `MyDrive/resources/ml-100k/ml-100k/u.data`.
2. Open `ml-100k-parameter-optimizer/ml-100k-parameter-optimizer.ipynb` in Colab with a GPU runtime.
3. Run all cells. Grid search (~125 configs) and 5-fold CV will execute sequentially.

### `genome-2021-movie-recommender`

1. Download [MovieLens Tag Genome 2021](https://grouplens.org/datasets/movielens/tag-genome-2021/) and place `ratings.json` and `metadata.json` files under `MyDrive/resources/genome_2021/movie_dataset_public_final/raw/`.
2. Open `genome-2021-movie-recommender/content-based-movie-recommender.ipynb` in Colab with a **High-RAM** runtime.
3. Run the preprocessing cell once. It cleans the 2.6M reviews and saves the result to `MyDrive/resources/processed_for_colab/cleaned_movie_docs.parquet`.
4. On subsequent runs, use the "New Setup" cell to load directly from the Parquet checkpoint.

---

## Citation

If you reference this work, please cite as:

@misc{aygun2025recommender,
  author       = {Aygün, Esranur},
  title        = {Recommender Systems: Collaborative Filtering and Content-Based Methods},
  year         = {2025},
  howpublished = {Graduate Coursework Repository},
  url          = {https://github.com/fukichime/Recommender-Systems}
}

---

### Datasets

Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets: History and Context.
ACM Transactions on Interactive Intelligent Systems, 5(4), 1–19.
https://doi.org/10.1145/2827872

Vig, J., Sen, S., & Riedl, J. (2012). The Tag Genome: Encoding Community Knowledge
to Support Novel Interaction. ACM Transactions on Interactive Intelligent Systems, 2(3).
https://doi.org/10.1145/2362394.2362395

