# Mall Customer Segmentation

Groups mall customers into actionable marketing segments based on age, income, and spending behavior — then trains a classifier on the resulting segments so new customers can be assigned instantly, without re-running clustering.

## Why this project

Customer segmentation is one of the most direct business applications of unsupervised learning — but a set of cluster *numbers* is useless to a marketing team on its own. This project goes a step further than a typical "K-Means on 3 features" tutorial:

- **Two independent clustering methods** (K-Means, cross-checked with hierarchical/agglomerative clustering) confirm the same 5-segment structure — one method alone could just be an artifact of the algorithm's assumptions.
- **Clusters are profiled into plain-language customer types**, not left as anonymous IDs.
- **Segments are turned into a reusable classifier** (SVM, Decision Tree) — the practical version of "how do I assign a *new* customer to a segment without re-clustering everyone."

## Project structure

```
├── mall_clustering.ipynb   # Full analysis: EDA → K-Means → hierarchical → profiling → classifier
├── Mall_Customers.csv      # Dataset (200 customers)
└── requirements.txt
```

## Dataset

[Mall Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) (Kaggle) — 200 customers, with age, gender, annual income, and a mall-assigned spending score.

## Approach

1. **EDA** — distribution of each variable, pairwise relationships (income vs. spending doesn't follow an obvious straight line, which is exactly why clustering helps).
2. **Preprocessing** — label-encode gender, standardize all features (required since K-Means uses Euclidean distance and unscaled income would dominate).
3. **K-Means** — elbow method to choose *k*, settles on 5 clusters.
4. **Hierarchical clustering** — an independent cross-check via dendrogram, confirming the same 5-cluster structure.
5. **Cluster profiling** — translating cluster IDs into actual customer descriptions (e.g. high income/low spenders vs. young high spenders).
6. **Segment → classifier** — SVM and Decision Tree trained to predict a customer's segment from raw features, so new customers can be classified without re-running clustering.

## Results

| Step | Result |
|---|---|
| Optimal clusters (elbow method) | k = 5 |
| Cross-check (hierarchical) | Confirms 5-cluster structure |
| Segment classifier — SVM | 92.5% accuracy |
| Segment classifier — Decision Tree | 90.0% accuracy |

## Running it locally

```bash
pip install -r requirements.txt
jupyter notebook mall_clustering.ipynb
```

## What I'd improve with more time

- Validate cluster quality with a silhouette score rather than relying on the elbow method alone.
- Try DBSCAN as a third clustering approach — it doesn't assume roughly spherical clusters the way K-Means does.
- Test the trained classifier on genuinely new/synthetic customer profiles, not just a held-out split of the same 200 rows.

## Author

Ans Tanveer Hashmi — BS Data Science, MNS University of Agriculture, Multan.
[LinkedIn] · [GitHub]
