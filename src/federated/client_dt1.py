# client_dt1.py

from project_1.src.federated.client import FlowerClient
import flwr as fl

fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(
        "../data/raw/DT1_log-timeline.csv"
    )
)