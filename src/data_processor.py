import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import joblib
import torch
import os

# Loading the data
raw_data = "data\\raw\\adult.csv"
raw_df = pd.read_csv(raw_data)
X = raw_df.drop(columns=["income"])
y = raw_df["income"]

# Splitting the data
X_train, X_holdout, y_train, y_holdout = train_test_split(
    X, y, test_size=0.5, random_state=42
)

# Printing out the shape
print("Train shape:", X_train.shape)
print("Holdout shape:", X_holdout.shape)

# Identifying numerical columns
numeric_cols = X_train.select_dtypes(include='number').columns.tolist()
print("Numerical Columns: ", numeric_cols)

# normalization
scaler = MinMaxScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_holdout[numeric_cols] = scaler.transform(X_holdout[numeric_cols])
print(X_train.columns)

# Identifying categorical columns
cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()
print("Categorical Columns: ", cat_cols)

# Encoder
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
ohe.set_output(transform="pandas")
encoded_cols= ohe.fit_transform(X_train[cat_cols])
encoded_cols_holdout = ohe.transform(X_holdout[cat_cols])
X_train = X_train.drop(columns=cat_cols).join(encoded_cols)
X_holdout = X_holdout.drop(columns=cat_cols).join(encoded_cols_holdout)
print(X_train.columns)

# Saving the models
filepath_scaler = 'data/processed/scaler.joblib'
filepath_ohe = 'data/processed/ohe.joblib'
os.makedirs(os.path.dirname(filepath_scaler), exist_ok=True)
joblib.dump(scaler, filepath_scaler)
os.makedirs(os.path.dirname(filepath_ohe), exist_ok=True)
joblib.dump(ohe, filepath_ohe)

# dataframe to tensors
X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
X_holdout_tensor = torch.tensor(X_holdout.to_numpy(), dtype=torch.float32)
torch.save(X_train_tensor, 'data/processed/X_train.pt')
torch.save(X_holdout_tensor, 'data/processed/X_holdout.pt')


