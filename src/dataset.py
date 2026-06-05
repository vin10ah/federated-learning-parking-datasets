import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class DemandDataset(Dataset):

    def __init__(
        self,
        timeline_path,
        window_size=12,
        interval="5min"
    ):

        df = pd.read_csv(timeline_path)

        df["datetime"] = pd.to_datetime(
            df["kst_date"].astype(str)
            + " "
            + df["kst_time"].astype(str)
        )

        # -------------------------
        # 차량 유입량 생성
        # -------------------------

        arrival = (
            df.groupby(
                pd.Grouper(
                    key="datetime",
                    freq=interval
                )
            )["car_id"]
            .nunique()
            .reset_index()
        )

        arrival.columns = [
            "datetime",
            "arrival_count"
        ]

        values = arrival["arrival_count"].values

        X = []
        y = []

        for i in range(
            len(values) - window_size
        ):

            X.append(
                values[i:i+window_size]
            )

            y.append(
                values[i+window_size]
            )

        self.X = np.array(
            X,
            dtype=np.float32
        )

        self.y = np.array(
            y,
            dtype=np.float32
        )

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return (
            self.X[idx],
            self.y[idx]
        )