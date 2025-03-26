"""
==========================================
Multimodal Losses for Cross-Modal Learning
==========================================

This example demonstrates the various multimodal losses available in kaira for
training models that work with multiple modalities (e.g., text-image, audio-video).

We'll cover:
- Contrastive Loss
- Triplet Loss
- InfoNCE Loss (Info Noise-Contrastive Estimation)
"""

# %%
# First, let's import the necessary modules
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from kaira.losses import LossRegistry

# %%
# Let's create some sample embeddings to simulate features from different modalities
def create_sample_embeddings(n_samples=100, n_dim=128):
    # Create anchor embeddings (e.g., image features)
    anchors = torch.randn(n_samples, n_dim)
    anchors = nn.functional.normalize(anchors, p=2, dim=1)
    
    # Create positive embeddings (similar to anchors)
    # Add small perturbations to anchors
    positives = anchors + 0.1 * torch.randn(n_samples, n_dim)
    positives = nn.functional.normalize(positives, p=2, dim=1)
    
    # Create negative embeddings (different from anchors)
    negatives = torch.randn(n_samples, n_dim)
    negatives = nn.functional.normalize(negatives, p=2, dim=1)
    
    # Create labels
    labels = torch.arange(n_samples)
    
    return anchors, positives, negatives, labels

# Create sample embeddings
anchors, positives, negatives, labels = create_sample_embeddings()

# %%
# Now let's compute different multimodal losses

# Contrastive Loss
contrastive_loss = LossRegistry.create('contrastiveloss', margin=0.5)
contrastive_value = contrastive_loss(anchors, positives, labels)
print(f'Contrastive Loss: {contrastive_value:.4f}')

# Triplet Loss
triplet_loss = LossRegistry.create('tripletloss', margin=0.3)
triplet_value = triplet_loss(anchors, positives, negatives)
print(f'Triplet Loss: {triplet_value:.4f}')

# InfoNCE Loss
infonce_loss = LossRegistry.create('infonceloss', temperature=0.07)  # Changed from 'infoNCEloss' to 'infonceloss'
infonce_value = infonce_loss(anchors, positives)
print(f'InfoNCE Loss: {infonce_value:.4f}')

# %%
# Let's visualize how these losses behave with different similarity values
def compute_similarity_losses(similarity):
    """Compute losses for a given cosine similarity value."""
    # Create vectors with specified cosine similarity and consistent dtype
    v1 = torch.tensor([[1.0, 0.0]], dtype=torch.float32)  # Explicitly set dtype
    v2 = torch.tensor([[similarity, np.sqrt(1 - similarity**2)]], dtype=torch.float32)  # Match dtype
    
    # Expand to batch
    v1_batch = v1.expand(10, 2)
    v2_batch = v2.expand(10, 2)
    
    # Compute losses
    losses = {
        'Contrastive': contrastive_loss(v1_batch, v2_batch).item(),
        'Triplet': triplet_loss(v1_batch, v2_batch, -v2_batch).item(),
        'InfoNCE': infonce_loss(v1_batch, v2_batch).item()
    }
    return losses

# Generate range of similarity values
similarities = np.linspace(-1, 1, 100)
loss_curves = {name: [] for name in ['Contrastive', 'Triplet', 'InfoNCE']}

for sim in similarities:
    losses = compute_similarity_losses(sim)
    for name, loss in losses.items():
        loss_curves[name].append(loss)

# %%
# Plot how losses vary with cosine similarity
plt.figure(figsize=(10, 6))
for name, losses in loss_curves.items():
    plt.plot(similarities, losses, label=name)

plt.xlabel('Cosine Similarity')
plt.ylabel('Loss Value')
plt.title('Loss Response to Embedding Similarity')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Let's examine the clustering behavior of these losses
def plot_embedding_clusters(embeddings, labels, title):
    # Use t-SNE for visualization
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings.detach().numpy())
    
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
               c=labels, cmap='tab10')
    plt.title(title)
    plt.colorbar(label='Class')

# Plot original embeddings
plt.figure(figsize=(15, 5))

plt.subplot(131)
plot_embedding_clusters(anchors, labels, 'Anchor Embeddings')

plt.subplot(132)
plot_embedding_clusters(positives, labels, 'Positive Embeddings')

plt.subplot(133)
plot_embedding_clusters(negatives, labels, 'Negative Embeddings')

plt.tight_layout()
plt.show()

# %%
# Let's also visualize the effect of the margin parameter in triplet loss
margins = [0.1, 0.3, 0.5, 1.0]
anchor_point = torch.tensor([[1.0, 0.0]])
theta = np.linspace(0, 2*np.pi, 100)
loss_values = {margin: [] for margin in margins}

for t in theta:
    point = torch.tensor([[np.cos(t), np.sin(t)]])
    for margin in margins:
        triplet_loss_margin = LossRegistry.create('tripletloss', margin=margin)
        loss = triplet_loss_margin(
            anchor_point.expand(10, 2),
            point.expand(10, 2),
            -point.expand(10, 2)
        ).item()
        loss_values[margin].append(loss)

# Plot loss values in polar coordinates
plt.figure(figsize=(10, 10))
ax = plt.subplot(111, projection='polar')
for margin, losses in loss_values.items():
    ax.plot(theta, losses, label=f'Margin={margin}')

plt.title('Triplet Loss Values Around Unit Circle')
plt.legend()
plt.show()

# %%
# This example demonstrates various losses used in multimodal learning:
#
# - Contrastive Loss brings similar embeddings closer while pushing dissimilar
#   ones apart, useful for tasks like face verification or image retrieval.
#
# - Triplet Loss ensures that an anchor is closer to a positive example than to
#   a negative example by a margin, commonly used in few-shot learning and
#   metric learning.
#
# - InfoNCE Loss is particularly effective for self-supervised learning and
#   contrastive representation learning, as it can handle multiple negative
#   examples efficiently.
#
# The visualizations show how these losses respond to different similarity
# values and how the margin parameter affects the triplet loss behavior.