# Recommender Systems

This repository contains my coursework, experiments, and projects focused on building and evaluating various recommendation algorithms.

## Project: Content-Based Movie Recommender System

**Folder:** `genome-2021-movie-recommender`
**Notebook:** `content-based-movie-recommender.ipynb`

### Introduction

The Content-Based Movie Recommender System is a machine learning project developed as a graduate-level assignment for a Recommender Systems course. It is designed to suggest movies to users based on the textual content of raw reviews. Unlike systems that rely on pre-existing metadata or tags, this project utilizes Natural Language Processing (NLP) to analyze over 2.6 million raw reviews from the MovieLens dataset. The primary objective is to predict user ratings and provide Top-N recommendations, evaluating performance against non-personalized baselines.

This project was developed as a graduate-level assignment for a Recommender Systems course.

### Key Features

* **NLP Data Pipeline:** Aggregates and cleans millions of raw text reviews (lowercasing, stopword removal) to generate unique content profiles for over 52,000 movies.
* **Optimized Data Loading:** Implements Parquet file checkpointing to efficiently load pre-processed text data, bypassing time-consuming cleaning steps during repeated runs.
* **TF-IDF Recommender:** Calculates movie similarity based on word importance using Term Frequency-Inverse Document Frequency and Cosine Similarity.
* **Binary/Jaccard Recommender:** Calculates movie similarity based on simple word presence using Binary Counts and Jaccard Similarity.
* **On-the-Fly Evaluation:** Utilizes a memory-efficient prediction loop that calculates similarity scores dynamically to prevent RAM crashes associated with massive matrices.
* **Performance Metrics:** Evaluates models using **Mean Absolute Error (MAE)** for rating accuracy and **Hit Ratio** for recommendation quality.

### Technologies Used

* **Python:** Primary language for analysis and modeling.
* **Google Colab Pro:** Cloud environment (High-RAM runtime is required for this dataset).
* **Pandas:** Data manipulation and aggregation of large datasets.
* **Scikit-learn:** Vectorization (TF-IDF, CountVectorizer), similarity calculations, and metrics.
* **NLTK:** Natural language preprocessing and stopword removal.
* **Google Drive:** Persistent storage for the dataset and checkpoint files.

### Dataset

This project utilizes the **MovieLens Tag Genome Dataset 2021**.

* **Source:** GroupLens
* **Download:** [genome_2021.zip](https://grouplens.org/datasets/movielens/tag-genome-2021/)
* **Storage:** The dataset was accessed directly from a personal Google Drive to enable persistent storage within Colab.

### Results & Performance

The models were evaluated on a sample of 5,000 ratings against two baselines.

| Model | MAE | RMSE | Hit Ratio @ 10 |
| :--- | :---: | :---: | :---: |
| **Random Baseline** | 1.5861 | 1.9565 | N/A |
| **Global Average Baseline** | 0.8355 | 1.0545 | N/A |
| **TF-IDF + Cosine (Model 1)** | 0.7607 | 0.9855 | 0.20% |
| **Binary + Jaccard (Model 2)** | **0.7593** | **0.9850** | *See Note* |

#### Findings
* **Accuracy:** Both content-based models significantly outperformed the Global Average baseline (MAE 0.8355). The **Binary + Jaccard model** achieved the best accuracy (MAE 0.7593), suggesting that simple word presence was a stronger signal than word frequency for this specific dataset.
* **Hit Ratio:** The TF-IDF model achieved a Hit Ratio of 0.20%.
* **Note:** The Jaccard Hit Ratio calculation was stopped due to excessive runtime (>2 hours) caused by dense array operations.
