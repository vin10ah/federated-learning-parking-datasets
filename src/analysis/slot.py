import pandas as pd

# 슬롯별 점유 로그 수

def slot_usage_table(timeline):

    slot_usage = (
        timeline["park_slot_id"]
        .dropna()
        .value_counts()
        .reset_index()
    )

    slot_usage.columns = [
        "slot_id",
        "usage_count"
    ]

    return slot_usage


# 슬롯별 점유율

def slot_occupancy_table(timeline):

    slot_usage = (
        timeline["park_slot_id"]
        .dropna()
        .value_counts()
    )

    df = (

        slot_usage
        / slot_usage.sum()

    ).reset_index()

    df.columns = [
        "slot_id",
        "occupancy_ratio"
    ]

    return df

# 슬롯별 이용 차량 수

def slot_vehicle_count_table(timeline):

    df = (

        timeline
        .dropna(
            subset=["park_slot_id"]
        )

        .groupby(
            "park_slot_id"
        )

        ["car_id"]

        .nunique()

        .reset_index()
    )

    df.columns = [
        "slot_id",
        "vehicle_count"
    ]

    return df.sort_values(
        "vehicle_count",
        ascending=False
    )

# 슬롯별 평균 체류시간

def duration_by_slot_table(final):

    df = (

        final

        .groupby(
            "slot_id"
        )

        ["duration"]

        .mean()

        .reset_index()
    )

    df.columns = [
        "slot_id",
        "avg_duration"
    ]

    return df.sort_values(
        "avg_duration",
        ascending=False
    )

# DT1~DT3 비교

def compare_slot_usage(datasets):

    rows = []

    for client_id, data in datasets.items():

        timeline = data["timeline"]

        slot_usage = (

            timeline["park_slot_id"]

            .dropna()

            .value_counts()
        )

        rows.append({

            "client":
                client_id,

            "slot_count":
                slot_usage.shape[0],

            "most_used_slot":
                slot_usage.idxmax(),

            "usage_count":
                int(
                    slot_usage.max()
                ),

            "avg_usage":
                round(
                    slot_usage.mean(),
                    2
                )
        })

    return pd.DataFrame(rows)


import matplotlib.pyplot as plt

 
# 슬롯 위치 및 사용량 표시

def plot_slot_layout_usage(timeline):

    slot_pos = (

        timeline
        .dropna(subset=["park_slot_id"])

        .groupby("park_slot_id")

        .agg(
            lat=("lat", "mean"),
            lon=("lon", "mean"),
            usage=("car_id", "count")
        )

        .reset_index()
    )

    plt.figure(figsize=(10, 8))

    plt.scatter(

        slot_pos["lon"],
        slot_pos["lat"],

        s=slot_pos["usage"] / 100,

        alpha=0.7
    )

    for _, row in slot_pos.iterrows():

        plt.text(

            row["lon"],
            row["lat"],

            row["park_slot_id"],

            fontsize=8
        )

    plt.title(
        "Parking Slot Layout + Usage"
    )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.grid(True)

    plt.show()




# 슬롯 위치 및 사용량 한번에 표시

def plot_all_slot_layouts(datasets):

    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(18, 6)
    )

    if len(datasets) == 1:
        axes = [axes]

    for ax, (client_id, data) in zip(
        axes,
        datasets.items()
    ):

        timeline = data["timeline"]

        slot_pos = (

            timeline
            .dropna(
                subset=["park_slot_id"]
            )

            .groupby(
                "park_slot_id"
            )

            .agg(
                lat=("lat", "mean"),
                lon=("lon", "mean"),
                usage=("car_id", "count")
            )

            .reset_index()
        )

        ax.scatter(
            slot_pos["lon"],
            slot_pos["lat"],
            s=slot_pos["usage"] / 100,
            alpha=0.7
        )

        for _, row in slot_pos.iterrows():

            ax.text(
                row["lon"],
                row["lat"],
                row["park_slot_id"],
                fontsize=8
            )

        ax.set_title(client_id)

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        ax.grid(True)

    plt.suptitle(
        "Parking Slot Layout Comparison",
        fontsize=14
    )

    plt.tight_layout()

    plt.show()