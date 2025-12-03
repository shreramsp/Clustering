# Document Clustering with LLM Embeddings

Clustering text documents using pretrained sentence transformer embeddings.

## Note

Notebook shows "Invalid" on GitHub due to metadata issues. See screenshot for full output.

## Files

- `Document_Clustering.ipynb` - Notebook (without execution)
- `Document_Clustering_LLM_Embeddings` - Implementation notebook (shows "Invalid")
- `full_execution_screenshot.png` - Complete run output
- `visualizations/` - Result visualizations

## Method

1. Load 20 newsgroups dataset (4 categories)
2. Generate embeddings using sentence-transformers
3. K-Means clustering on embeddings
4. Evaluate with ARI (~0.5-0.7) and Silhouette Score


