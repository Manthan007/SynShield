import sys
import os
# Dynamically add the current directory and parent directory to Python's search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from src.tab_ddpm import TabularMLP
from torch.optim import Adam
import torch.nn as nn
from opacus import PrivacyEngine
from tab_ddpm import q_sample

# Data Loading
X_train_tensor = torch.load("D:\\GenAI-Projects\\SynShield\\data\\processed\\X_train.pt")
X_train_tensor_dataset = TensorDataset(X_train_tensor)
loader = DataLoader(X_train_tensor_dataset, batch_size=128, shuffle=True)

# Model & Optimizer setup
model = TabularMLP(num_features=X_train_tensor.shape[1])
optimizer = Adam(model.parameters(), lr=1e-3)

# Opacus setup
engine = PrivacyEngine()
model, optimizer, loader = engine.make_private(
    module=model, 
    optimizer=optimizer, 
    data_loader=loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0)

# Training Loop
epochs = 50
criterion = nn.MSELoss()
for epoch in range(epochs):
    # setting model to training mode
    model.train()

    for batch in loader:
        x_0 = batch[0]
        batch_size = x_0.shape[0]

        t = torch.randint(low=0, high=1000, size=(batch_size,))
        x_t, true_noise = q_sample(x_0, t)
        predicted_noise = model(x_t, t)
        loss = criterion(predicted_noise, true_noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Calculate epsilon
    epsilon = engine.get_epsilon(delta=1e-5)
    print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | ε: {epsilon:.2f}")

# saving the model
torch.save(model.state_dict(), "data/processed/dp_diffusion.pt")
