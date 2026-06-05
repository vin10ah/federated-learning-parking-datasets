import pandas as pd

# 차량 최초 등장 시간

def build_arrival_series(
    timeline,
    freq="5min"
):

    first_seen = (

        timeline
        .groupby("car_id")
        ["datetime"]
        .min()
        .reset_index()
    )

    arrival = (

        first_seen
        .set_index("datetime")
        .resample(freq)
        .size()
    )

    arrival.name = "arrival_count"

    return arrival


# 통계

def demand_stats(
    timeline,
    freq="5min"
):

    arrival = (
        build_arrival_series(
            timeline,
            freq
        )
    )

    return (
        arrival
        .describe()
        .to_frame()
    )


# 피크 시간

def peak_demand(
    timeline,
    freq="5min"
):

    arrival = (
        build_arrival_series(
            timeline,
            freq
        )
    )

    return arrival.idxmax(), arrival.max()


import pandas as pd

# 전체 비교

def demand_stats_all(
    datasets,
    freq="5min"
):

    rows = []

    for client_id, data in datasets.items():

        timeline = data["timeline"]

        first_seen = (

            timeline
            .groupby("car_id")
            ["datetime"]
            .min()
            .reset_index()
        )

        arrival = (

            first_seen
            .set_index("datetime")
            .resample(freq)
            .size()
        )

        rows.append({

            "client":
                client_id,

            "avg_arrival":
                round(
                    arrival.mean(),
                    2
                ),

            "peak_arrival":
                int(
                    arrival.max()
                ),

            "peak_time":
                arrival.idxmax()
        })

    return pd.DataFrame(rows)