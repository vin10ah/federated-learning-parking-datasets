import flwr as fl

import torch

from project_1.src.data.dataset import DemandDataset
from project_1.src.models.model import DemandMLP

from torch.utils.data import DataLoader


def get_parameters(model):

    return [
        val.cpu().numpy()
        for _, val
        in model.state_dict().items()
    ]


def set_parameters(
    model,
    parameters
):

    params_dict = zip(
        model.state_dict().keys(),
        parameters
    )

    state_dict = {
        k: torch.tensor(v)
        for k, v in params_dict
    }

    model.load_state_dict(
        state_dict,
        strict=True
    )


class FlowerClient(
    fl.client.NumPyClient
):

    def __init__(
        self,
        data_path
    ):

        self.model = DemandMLP()

        dataset = DemandDataset(
            data_path
        )

        print(f"Dataset Size: {len(dataset)}")

        self.loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False
        )

    def get_parameters(
        self,
        config
    ):

        return get_parameters(
            self.model
        )

    def fit(
        self,
        parameters,
        config
    ):

        set_parameters(
            self.model,
            parameters
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-3
        )

        criterion = (
            torch.nn.MSELoss()
        )

        self.model.train()

        for _ in range(1):

            for x, y in self.loader:

                pred = self.model(x)

                loss = criterion(
                    pred.squeeze(),
                    y
                )

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                print(f"Loss: {loss.item():.4f}")

                print(self.model.net[0].weight[0][:5])


        return (
            get_parameters(
                self.model
            ),
            len(self.loader.dataset),
            {}
        )

    def evaluate(
        self,
        parameters,
        config
    ):

        return (
            0.0,
            len(self.loader.dataset),
            {}
        )