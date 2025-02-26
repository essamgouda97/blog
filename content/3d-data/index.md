---
title: 'Vibe Coding with 3D Models'
date: '2025-02-26'
excerpt: 'I trained a model that embeds 3D CAD models into a vector database.'
---

Since *vibe coding* is now a well-established term, I decided to try it on another form of data. I've worked extensively with images, audio, and text, but I had never trained a network on 3D models. Our life exists in 3D, so it only makes sense to understand how to work with 3D data. 🫡  

## Task: Train a Model That’s Good at Embedding 3D Models

I started with the [ModelNet10 dataset](https://www.kaggle.com/datasets/balraj98/modelnet10-princeton-3d-object-dataset/data) and implemented [PointNet](https://arxiv.org/abs/1612.00593), a straightforward neural network that accepts unstructured point data and is suitable for classification and segmentation tasks. However, what I really wanted was **"3D models → embeddings"** (fancy term for *a list of numbers that helps computers understand stuff*). These embeddings are stored in PointNet’s global feature layer (marked green).  

![PointNet Architecture](/content/3d-data/static/arch.png)  

## Data

First off, the data comes in `.off` files, a format that represents faces and vertices in a human-readable way. Thanks to this [Kaggle notebook](https://www.kaggle.com/code/balraj98/pointnet-for-3d-object-classification-pytorch/notebook), I was able to borrow some code to read the data. In summary:  

- **Faces & Vertices:** A vertex is a point in 3D space, and a face is the shape created when those points are connected.  
- **Mesh:** A structured shape, e.g., a **Triangular Mesh** (a face that looks like a triangle). Below is a **visual representation** of a bathtub read from a `.off` file:  

  ![Bathtub Mesh Visualization](/content/3d-data/static/bathtub_mesh.png)  

- Once we have the meshes, we need to **sample** them. Since I’m using PointNet, I sample the mesh into **points!**  

  ![Point Cloud Visualization](/content/3d-data/static/bathtub_points.png)  

## Self-Supervised Training with NT-Xent Loss  

In the real world, **data is unlabeled** cuz humans are messy. To try and mimic the real world I picked NT-Xent Loss that is usually used in a self-supervised way.

### What is NT-Xent Loss?  

NT-Xent stands for **Normalized Temperature-scaled Cross Entropy Loss**, and it’s a key component in **contrastive learning**. It helps the model learn **meaningful representations** by ensuring that similar data points (positive pairs) have embeddings that are close together, while dissimilar ones (negative pairs) are pushed apart.  

### How Does It Work?  

1. **Data Pairing Using Labels:**  
   - The *anchor* and *positive* come from the same class.  
   - The *negative* comes from a different class.  

2. **Embedding Similarity:**  
   - We compute the **cosine similarity** between all embeddings in a batch.  
   - The similarity scores are scaled using a **temperature parameter (τ)**, which controls how much we want to separate positive pairs from negative pairs.  

3. **Contrastive Loss Calculation:**  
   - The loss function ensures that the similarity between *positive pairs* is maximized, while the similarity with *negative pairs* is minimized:  

   $$
   L = -\log \frac{\exp(sim(z_i, z_j)/\tau)}{\sum_{k=1}^{N} \exp(sim(z_i, z_k)/\tau)}
   $$  

### How I Selected Positive and Negative Samples  

I **cheated** and used the class labels to construct *positive*, *negative*, and *anchor* samples instead of relying on augmentations.  

Since the dataset is labeled, I leveraged the existing class labels for contrastive learning:

- **Anchor:** A randomly chosen 3D model from a specific class.  
- **Positive:** Another sample from the same class.  
- **Negative:** A sample from a different class.  

If the dataset were **unlabeled**, I would have needed to:

1. Augment the same sample to create positives.  
2. Select a random 3D model as a negative.  
3. Ensure negatives were not too similar to the anchor by using feature distance constraints.

## Training and Results  

Training commenced, convergence happened, validation loss decreased—**good enough for the weekend.** Next.

## Adding PCA for Visualization  

To **visualize** how the model organizes 3D embeddings over time, I applied **Principal Component Analysis (PCA)** and plotted the clusters at:  

- **Epoch 0:** Initial embeddings, mostly scattered.  
- **Epoch 100:** Well-separated clusters, indicating that the model learned meaningful representations.  

![PCA Visualization at Epoch 0 vs. Epoch 100](#)

## Shoutout to  

- My **Kaggle** and **Hugging Face** homies  
- **ChatGPT** and **Cursor**  
- **Math** for being awesome and keeping it real  

## Challenges and Roadmap  

- I want to use a **better dataset**—finding a reliable 3D dataset is difficult. I requested access to [PartNet](https://huggingface.co/datasets/ShapeNet/PartNet-archive) on Hugging Face and somehow got it. Any suggestions for other datasets?  
- Can I use **better loss functions**? I'm thinking about experimenting with **Triplet Loss** and **Barlow Twins Loss** to see if they improve embedding quality.  
- Next, I want to **populate a vector database** (FAISS or Milvus) and perform searches on it.  
- Is there a **better architecture**? A **better representation** of 3D models other than point clouds?

More to come. Cheers 🥂
