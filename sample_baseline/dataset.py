import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_data(npz_path, batch_size=32):

    data = np.load(npz_path)

    X = data['x']
    y = data['y']

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X, y)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return loader, X.shape[-1]