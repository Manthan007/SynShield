import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

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
# print("Train shape:", X_train.shape)
# print("Holdout shape:", X_holdout.shape)

# Identifying numerical columns
numeric_cols = X_train.select_dtypes(include='number').columns.tolist()
# print("Numerical Columns: ", numeric_cols)

# normalization
scaler = MinMaxScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
print(X_train.columns)

# Identifying categorical columns
cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()
print("Categorical Columns: ", cat_cols)

# Encoder
ohe = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
ohe.set_output(transform="pandas")
encoded_cols= ohe.fit_transform(X_train[cat_cols])
X_train = X_train.drop(columns=cat_cols).join(encoded_cols)
print(X_train.columns)


