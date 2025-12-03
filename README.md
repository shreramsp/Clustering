# Clustering Algorithms Implementation

A comprehensive collection of clustering algorithms implemented in Python, demonstrating both classical and modern approaches including LLM-based embeddings for multimodal data.

## 📚 Assignment Overview

Implementation of various clustering techniques on diverse datasets, exploring unsupervised learning methods from traditional algorithms to state-of-the-art multimodal embeddings.

## 📂 Repository Structure

### **A. K-Means Clustering from Scratch**
`K_Means_Clustering_from_Scratch.ipynb`
- Pure Python implementation of K-Means algorithm using NumPy
- Dataset: Synthetic blobs (make_blobs)
- Demonstrates initialization, assignment, and update steps
- Metrics: Silhouette Score, Inertia

### **B. Hierarchical Clustering**
`Hierarchical_Clustering.ipynb`
- Agglomerative clustering with multiple linkage methods (Ward, Complete, Average, Single)
- Dataset: Iris (scikit-learn)
- Dendrogram visualization and optimal cluster identification
- Metrics: Silhouette Score, Adjusted Rand Index

### **C. Gaussian Mixture Models (GMM)**
`Gaussian_Mixture_Models_(GMM)_Clustering.ipynb`
- Probabilistic clustering with soft assignments
- Dataset: Synthetic overlapping clusters (make_blobs)
- Model selection using BIC/AIC
- Probability contour visualization
- Metrics: Silhouette Score, Log-Likelihood, BIC, AIC

### **D. DBSCAN Clustering with PyCaret**
`DBSCAN_Clustering_with_PyCaret.ipynb`
- Density-based clustering for arbitrary shapes and outlier detection
- Dataset: Synthetic moon shapes with noise (make_moons)
- Automated ML workflow using PyCaret
- Identifies clusters without pre-specifying number
- Metrics: Silhouette Score (excluding noise), cluster count

### **E. Anomaly Detection using PyOD**
`Anomaly_Detection_using_PyOD.ipynb`
- Multiple outlier detection algorithms (Isolation Forest, LOF, KNN)
- Dataset: Synthetic data with 10% contamination (PyOD generator)
- Comparative analysis of detection methods
- Anomaly score distribution analysis
- Metrics: Precision, Recall, F1-Score, Confusion Matrix

### **F. Time-Series Clustering**
`TimeSeries_Clustering.ipynb`
- Clustering temporal patterns using Dynamic Time Warping (DTW)
- Dataset: UCR Trace (electrical power demand)
- DTW vs Euclidean distance comparison
- TSLearn library for time-series specific methods
- Metrics: Adjusted Rand Index, Inertia

### **G. Document Clustering with LLM Embeddings**
`Document_Clustering_LLM_Embeddings/`
- Text clustering using sentence transformer embeddings
- Dataset: 20 Newsgroups (4 categories)
- Pretrained model: all-MiniLM-L6-v2
- UMAP dimensionality reduction for visualization
- Semantic similarity beyond keyword matching
- Metrics: Adjusted Rand Index (~0.5-0.7), Silhouette Score

### **H. Image Clustering with ImageBind**
`Image_Clustering_ImageBind.ipynb`
- Image clustering using Meta's multimodal embeddings
- Dataset: Fashion-MNIST (10 categories, 1000 samples)
- ImageBind model for vision embeddings
- Grayscale to RGB preprocessing
- UMAP visualization and cluster purity analysis
- Metrics: Adjusted Rand Index (~0.5), Silhouette Score (~0.12)

### **I. Audio Clustering with ImageBind**
`Audio_Clustering_ImageBind.ipynb`
- Environmental sound clustering with multimodal embeddings
- Dataset: ESC-50 subset (10 categories, 200 samples)
- Categories: Animals, nature, indoor, urban sounds
- Spectrogram visualization
- Acoustic similarity analysis
- Metrics: Adjusted Rand Index, Silhouette Score

## 🛠️ Technologies Used

- **Core Libraries**: NumPy, Pandas, Matplotlib, Seaborn
- **Clustering**: scikit-learn, scipy, PyCaret
- **Anomaly Detection**: PyOD
- **Time-Series**: TSLearn
- **NLP**: sentence-transformers, transformers
- **Audio/Vision**: ImageBind (Meta), librosa, torchaudio
- **Dimensionality Reduction**: UMAP

## 📊 Key Concepts Demonstrated

- **Traditional Clustering**: K-Means, Hierarchical, GMM, DBSCAN
- **Specialized Applications**: Anomaly detection, time-series, multimodal
- **Modern Embeddings**: Pretrained LLM embeddings for text, images, and audio
- **Evaluation Metrics**: Silhouette Score, ARI, BIC/AIC, precision/recall
- **Visualization**: UMAP, dendrograms, confusion matrices, spectrograms

## 🚀 Running the Notebooks

All notebooks are designed for Google Colab with GPU runtime (recommended for H & I).

1. Upload notebook to Google Colab
2. Set runtime to GPU (for ImageBind notebooks)
3. Run all cells
4. Most notebooks auto-download required datasets

## 📌 Notes

- Notebooks G, H, I may show "Invalid" on GitHub due to metadata - see execution screenshots in folders
- ImageBind notebooks require GPU for optimal performance (~10-15 min on A100)
- Results may vary slightly due to random initialization in clustering algorithms

## 📖 Assignment Context

This repository fulfills a Data Mining course assignment on clustering algorithms, demonstrating progression from classical methods to modern multimodal LLM-based approaches.

---

**Course**: Data Mining  
**Topic**: Clustering Algorithms  
**Techniques**: 9 different clustering approaches across diverse data types
