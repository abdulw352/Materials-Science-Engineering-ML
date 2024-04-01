import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Data preprocessing
def preprocess_data(data_dir):
    """Preprocess data from Matmatch.com."""
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    all_data = []

    for file in data_files:
        file_path = os.path.join(data_dir, file)
        data = pd.read_csv(file_path)
        data = data[["formula", "space_group", "stability"]]
        all_data.append(data)

    combined_data = pd.concat(all_data, ignore_index=True)
    combined_data.dropna(inplace=True)
    combined_data.reset_index(drop=True, inplace=True)

    X = combined_data[["formula", "space_group"]]
    y = combined_data["stability"]

    return X, y

# Custom dataset class
class MaterialsDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        formula = self.X.iloc[idx]["formula"]
        space_group = self.X.iloc[idx]["space_group"]
        stability = self.y.iloc[idx]

        # Convert formula to composition object
        composition = Composition(formula)

        # Convert space group to one-hot encoding
        space_group_one_hot = torch.zeros(230)  # Assuming 230 space groups
        space_group_one_hot[space_group - 1] = 1

        # Convert stability to a tensor
        stability_tensor = torch.tensor([stability], dtype=torch.float32)

        return composition, space_group_one_hot, stability_tensor

# Model architecture
class MaterialsModel(nn.Module):
    def __init__(self):
        super(MaterialsModel, self).__init__()
        self.composition_embedding = nn.Embedding(num_embeddings=100, embedding_dim=32)
        self.space_group_embedding = nn.Linear(230, 16)
        self.fc1 = nn.Linear(32 + 16, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, composition, space_group):
        composition_embed = self.composition_embedding(composition)
        space_group_embed = self.space_group_embedding(space_group)
        x = torch.cat((composition_embed, space_group_embed), dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Training setup
data_dir = "path/to/matmatch/data"
X, y = preprocess_data(data_dir)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

train_dataset = MaterialsDataset(X_train, y_train)
val_dataset = MaterialsDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = MaterialsModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for composition, space_group, stability in train_loader:
        optimizer.zero_grad()
        outputs = model(composition, space_group)
        loss = criterion(outputs, stability)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * stability.size(0)

    model.eval()
    val_loss = 0.0
    val_predictions = []
    val_targets = []
    with torch.no_grad():
        for composition, space_group, stability in val_loader:
            outputs = model(composition, space_group)
            loss = criterion(outputs, stability)
            val_loss += loss.item() * stability.size(0)
            val_predictions.extend(outputs.squeeze().tolist())
            val_targets.extend(stability.squeeze().tolist())

    train_loss /= len(train_dataset)
    val_loss /= len(val_dataset)
    val_rmse = mean_squared_error(val_targets, val_predictions, squared=False)
    val_r2 = r2_score(val_targets, val_predictions)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}, Val R2: {val_r2:.4f}")

# Save the trained model
torch.save(model.state_dict(), "materials_model.pth")
