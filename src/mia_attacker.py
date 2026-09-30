import torch
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Load data
X_train = torch.load("data\\processed\\X_train.pt")
X_holdout = torch.load("data\\processed\\X_holdout.pt")
X_synthetic = torch.load("data\\synthetic\\x_synthetic.pt")

# Subsampling
train_sample = X_train[:500].numpy()
holdout_sample = X_holdout[:500].numpy()
synthetic_sample = X_synthetic[:1000].numpy()

# Nearest Neigbour 
nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
nn.fit(synthetic_sample)

# Query both against the synthetic dataset
dist_train, _ = nn.kneighbors(train_sample)
dist_holdout, _ = nn.kneighbors(holdout_sample)

# Evaluation
mean_train_dist = np.mean(dist_train)
mean_holdout_dist = np.mean(dist_holdout)

print("\n--- ATTACK RESULTS ---")
print(f"Mean Distance (Train -> Synthetic):   {mean_train_dist:.4f}")
print(f"Mean Distance (Holdout -> Synthetic): {mean_holdout_dist:.4f}")

# Calculate Attacker Accuracy
# Combine all distances to find the global median
all_distances = np.concatenate([dist_train, dist_holdout])
median_dist = np.median(all_distances)

# The attacker guesses "Training Data" if the distance is less than the median
true_positives = np.sum(dist_train < median_dist)
true_negatives = np.sum(dist_holdout >= median_dist)

total_records = len(train_sample) + len(holdout_sample)
accuracy = (true_positives + true_negatives) / total_records

print(f"\nAttacker Accuracy: {accuracy * 100:.2f}%")
