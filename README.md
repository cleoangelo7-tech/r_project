# Content Recommendation System for a Free Streaming Platform

**Cleo Angelo | March 2026**

## Overview

This project builds and compares three recommendation approaches using the MovieLens Latest dataset (~33M ratings, sampled to 1M), framed around the specific challenges of a **free, ad-supported streaming platform**.

Free platforms like Tubi operate differently from subscription services: revenue comes from ad impressions, which means **more watch time = more revenue**. This changes what "good recommendations" means — it's not just about predicting ratings accurately, but about maximizing engagement, surfacing long-tail content, and keeping users in-session longer.

## Approaches

### 1. Content-Based Filtering
TF-IDF vectorization on genre and user-generated tag features, with cosine similarity to find similar movies.

- **Best for:** Cold-start users, long-tail content discovery
- **How it works:** Builds feature vectors from genres/tags, computes pairwise similarity across the catalog
- **Tradeoff:** Reliable and transparent, but creates genre "echo chambers" — it can't recommend across taste boundaries

### 2. Collaborative Filtering (KNN)
Item-based K-nearest neighbors on a sparse user-item matrix, using cosine distance to find movies with similar rating patterns.

- **Best for:** Users with moderate history (20+ ratings)
- **How it works:** Finds movies that were rated similarly by overlapping users, then predicts ratings via weighted neighbor average
- **Tradeoff:** Captures cross-genre taste patterns that content-based misses, but computationally expensive and biased toward popular titles

### 3. Matrix Factorization (SVD)
Truncated SVD on the mean-centered user-item matrix to extract latent preference factors.

- **Best for:** Scalable personalization for established users
- **How it works:** Decomposes the user-item matrix into low-rank latent factors, then reconstructs predicted ratings. Mean-centering per user corrects for rating scale differences.
- **Tradeoff:** Best accuracy-to-scalability ratio, but less interpretable than the other approaches

## Key Findings

### From EDA
- The user-item matrix is **extremely sparse** — the vast majority of user-movie combinations have no rating, making collaborative filtering challenging without filtering to active users.
- Ratings follow a **long-tail distribution**: a small set of blockbusters dominates, while most movies have very few ratings. For a free platform, this long tail represents untapped ad revenue.
- There's a **positive rating bias** — users rate generously (most ratings are 3+), which means distinguishing "good" from "great" requires sensitivity to small differences. In production, implicit signals like watch completion would be more informative.
- **Genre popularity doesn't align with genre quality.** Drama and Comedy dominate the catalog, but niche genres like Film-Noir and Documentary have higher average ratings. Personalized recommendations can bridge this gap.

### From Modeling
- **Content-based** produces genre-coherent recommendations and works for every movie in the catalog, including obscure titles with zero ratings.
- **KNN** generates more diverse, cross-genre recommendations but skews toward popular titles (popularity bias).
- **SVD** achieves competitive or better RMSE than KNN while being significantly faster at inference time — a critical advantage for production systems serving millions of users.

## Product Recommendations

For a free, ad-supported streaming platform, I'd recommend a **phased hybrid approach**:

**Phase 1 — Hybrid baseline:** Content-based for cold-start users (< 20 interactions), SVD for established users. SVD over KNN because it scales linearly and is O(1) at inference.

**Phase 2 — Engagement optimization:** Replace RMSE with business metrics (CTR, session duration, completion rate). A/B test the hybrid against baselines. Add diversity constraints to recommendation slates.

**Phase 3 — Advanced:** Incorporate temporal signals (time-of-day, recency weighting), contextual features (device type), sequential models (predict next watch), and migrate from explicit ratings to implicit feedback (watch duration).

### Key Tradeoffs

| Decision | Recommendation | Reasoning |
|----------|---------------|-----------|
| Accuracy vs. Diversity | Blend with diversity constraints | Pure accuracy creates filter bubbles |
| Popular vs. Niche | Balance both | Popular builds trust, niche drives catalog utilization |
| Explicit vs. Implicit signals | Implicit (watch time) | More scalable, less friction, captures passive engagement |
| Model complexity | Start simple, iterate | Ship SVD + content-based, optimize with A/B tests |

### Beyond RMSE

For streaming, these metrics matter as much as prediction accuracy:
- **Coverage** — % of catalog surfaced. Every unwatched title is lost ad revenue.
- **Diversity** — variety within recommendation slates. Prevents fatigue, extends sessions.
- **Novelty** — helps users discover content they'd never search for.
- **Serendipity** — surprising AND relevant. Builds trust in the algorithm.

## Dataset

[MovieLens Latest Full](https://grouplens.org/datasets/movielens/latest/) — ~33M ratings from 330K users on 86K movies (GroupLens Research). Sampled to 1M ratings for training speed with reproducible seed (random_state=42).

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/streaming-recommendation-system.git
cd streaming-recommendation-system
pip install -r requirements.txt
```

Download [ml-latest.zip](https://grouplens.org/datasets/movielens/latest/), unzip into the project directory so you have `ml-latest/ratings.csv`, `ml-latest/movies.csv`, `ml-latest/tags.csv`.

```bash
jupyter notebook tubi_recommendation_system.ipynb
```

The notebook runs top-to-bottom sequentially.

## Repo Structure

```
├── README.md
├── requirements.txt
├── tubi_recommendation_system.ipynb
└── ml-latest/
    ├── ratings.csv
    ├── movies.csv
    ├── tags.csv
    └── links.csv
```

## Limitations

- MovieLens uses explicit ratings; streaming platforms rely on implicit feedback (watch time, completion)
- Sampled to 1M ratings for speed — production would use full dataset
- No temporal or contextual features in models
- Content features limited to genres/tags — richer metadata (cast, director, mood) would improve quality
- Evaluated on RMSE, but true objective is engagement (different optimization target)

## Author

**Cleo Angelo** — All code and analysis is original work by me.

## Acknowledgments

- [MovieLens](https://grouplens.org/datasets/movielens/) by GroupLens Research
- scikit-learn, scipy, pandas, matplotlib, seaborn
