import torch

from dataset import DemandDataset
from model import DemandMLP

from torch.utils.data import DataLoader


dataset = DemandDataset(
    "../data/DT1_log-timeline.csv"
)

n = len(dataset)

train_size = int(n * 0.8)
val_size = int(n * 0.1)
test_size = n - train_size - val_size

train_set = torch.utils.data.Subset(
    dataset,
    range(0, train_size)
)

val_set = torch.utils.data.Subset(
    dataset,
    range(train_size,
          train_size+val_size)
)

test_set = torch.utils.data.Subset(
    dataset,
    range(train_size+val_size,
          n)
)

train_loader = DataLoader(
    train_set,
    batch_size=32,
    shuffle=False
)

model = DemandMLP()

criterion = torch.nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

for epoch in range(10):

    model.train()

    total_loss = 0

    for x, y in train_loader:

        pred = model(x)

        loss = criterion(
            pred.squeeze(),
            y
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch} "
        f"Loss {total_loss:.4f}"
    )