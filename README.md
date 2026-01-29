# ⚡ Parallel RBM Optimization (CUDA & TensorFlow)

**📅 Project Date:** Fall 2023
**🛠️ Tech Stack:** Python, TensorFlow (Graph Mode), CUDA, Pandas, NumPy
**📉 Application:** Collaborative Filtering Recommender System (MovieLens Dataset)

### 📖 Project Overview
Restricted Boltzmann Machines (RBMs) are energy-based neural networks used for unsupervised feature learning. Training them on large datasets (like MovieLens) is computationally expensive due to the iterative **Gibbs Sampling** process.

This project implements a parallelized RBM training loop using **TensorFlow** and **CUDA** to accelerate the Contrastive Divergence (CD-k) algorithm. The goal was to benchmark GPU vs. CPU performance for matrix-heavy operations.

### 🏗️ Technical Implementation
The project [source code](./RBM_MovieLens_Recommender.ipynb) implements:
1.  **Graph-Based Computation:** Utilizes TensorFlow's static graph (`tf.disable_v2_behavior`) for optimized tensor allocation.
2.  **Data Parallelism:** Batched training updates to maximize GPU core utilization.
3.  **Recommendation Engine:** Generates user scores by reconstructing the visible layer from hidden feature activations.

### 📊 Performance Results
* **Baseline:** Serial Python implementation (CPU).
* **Optimized:** Parallel implementation (NVIDIA CUDA).
* **Speedup:** Achieved **1.12x reduction** in training latency.
* **Accuracy:** Successfully recommended movies by identifying latent user preference clusters.

### 📸 Training Workflow
*(See the [Notebook](./RBM_MovieLens_Recommender.ipynb) for full execution logs)*
1.  **Preprocessing:** Merging `movies.dat` and `ratings.dat` into a sparse user-item matrix.
2.  **Training:** 50 Epochs with Mean Squared Error (MSE) loss tracking.
3.  **Inference:** Sorting predicted ratings to output top-N recommendations.

### 📄 Documentation
* [**Technical Report (PDF)**](./Parallelizing%20the%20Restricted%20Boltzmann%20Machine%20report.pdf) - Detailed math and performance analysis.
* [**Architecture Slides (PPTX)**](./HPC%20Presentation(RBM).pptx) - System diagrams and speedup charts.
