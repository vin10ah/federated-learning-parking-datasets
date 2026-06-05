import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import os

from model import LSTM, SimpleMLP 
from dataset import load_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class FlowerClient(fl.client.NumPyClient):

    def __init__(self, client_path):

        self.train_path = os.path.join(client_path, "train.npz")
        self.valid_path = os.path.join(client_path, "val.npz")

        self.train_loader, input_dim = load_data(self.train_path)
        self.valid_loader, input_dim = load_data(self.valid_path)

        self.model = LSTM(input_dim).to(DEVICE)

        self.criterion = nn.MSELoss()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.001
        )

    # 서버 -> 클라이언트 weight 전달
    def get_parameters(self, config):

        return [
            val.cpu().numpy()
            for _, val in self.model.state_dict().items()
        ]

    # 클라이언트 -> 모델 weight 적용
    def set_parameters(self, parameters):

        params_dict = zip(
            self.model.state_dict().keys(),
            parameters
        )

        state_dict = {
            k: torch.tensor(v)
            for k, v in params_dict
        }

        self.model.load_state_dict(state_dict, strict=True)

    # 로컬 학습
    def fit(self, parameters, config):

        self.set_parameters(parameters)  # 서버 -> 클라이언트 weight 적용

        self.model.train()

        for epoch in range(3):

            for X, y in self.train_loader:

                X = X.to(DEVICE)
                y = y.to(DEVICE)

                pred = self.model(X)

                loss = self.criterion(pred, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return (
            self.get_parameters(config={}),
            len(self.train_loader.dataset),
            {}
        )

    # 평가
    def evaluate(self, parameters, config):

        self.set_parameters(parameters)

        self.model.eval()

        total_loss = 0

        with torch.no_grad():

            for X, y in self.valid_loader:

                X = X.to(DEVICE)
                y = y.to(DEVICE)

                pred = self.model(X)

                loss = self.criterion(pred, y)

                total_loss += loss.item()

        return (
            float(total_loss),
            len(self.train_loader.dataset),
            {}
        )


if __name__ == "__main__":

    import sys

    train_npz = sys.argv[1] # 스크립트 실행 시 첫번째 명령줄 인자

    client = FlowerClient(train_npz)


    # flower 서버 접속 -> 현재 global model 받아서 로컬 학습 수행 -> weight 업로드
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client
    )