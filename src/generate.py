import os
import sys

# Dynamically add the current directory and parent directory to Python's search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch
from tab_ddpm import TabularMLP
from torch.utils.data import TensorDataset
import joblib
import pandas as pd


# Data AND Model Loading
X_train_tensor = torch.load("D:\\GenAI-Projects\\SynShield\\data\\processed\\X_train.pt")
X_train_tensor_dataset = TensorDataset(X_train_tensor)
model = TabularMLP(num_features=X_train_tensor.shape[1])
trained_model = model.load_state_dict(torch.load("data/processed/dp_diffusion.pt"), strict=False)

scaler = joblib.load("data\\processed\\scaler.joblib")
ohe = joblib.load("data\\processed\\ohe.joblib")

# setup parameters
beta = torch.linspace(start=(10)**(-4), end=0.02, steps=1000)
alpha = 1 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

# Genration Loop
model.eval()
x = torch.randn((10000, X_train_tensor.shape[1]))
with torch.no_grad():
    for i in reversed(range(1000)):
        t_batch = torch.full((10000,), i)
        predicted_noise = model(x, t_batch)

        if i > 0:
            z = torch.randn_like(x)
        else:
            z = 0

        x = 1/torch.sqrt(alpha[i]) * (x - ((1-alpha[i])/torch.sqrt(1-alpha_bar[i])) * predicted_noise) + torch.sqrt(beta[i]) * z


# Converting into numpy array
x_np = x.numpy()
numerical_synthetic = x_np[:, :6]
categorical_synthetic = x_np[:, 6:]

# Creating the dataframe
num_df = scaler.inverse_transform(numerical_synthetic) 
cat_df = ohe.inverse_transform(categorical_synthetic)

# numpy arrays converted to dataframes
num_df_converted = pd.DataFrame(num_df)
cat_df_converted = pd.DataFrame(cat_df)
synthetic_adult = pd.concat([num_df_converted, cat_df_converted], axis=1)

# Saving the csv
os.makedirs("data/synthetic", exist_ok=True)
synthetic_adult.to_csv("data/synthetic/synthetic_adult.csv")
# Save the raw tensor for the Attacker
torch.save(x, "data/synthetic/x_synthetic.pt")