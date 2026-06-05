import pandas as pd


def traffic_stats(timeline):

    result = {}

    result["total_cars"] = (
        timeline["car_id"]
        .nunique()
    )

    result["total_rows"] = (
        len(timeline)
    )

    result["start_time"] = (
        timeline["datetime"]
        .min()
    )

    result["end_time"] = (
        timeline["datetime"]
        .max()
    )

    result["duration_hours"] = round(
        (
            timeline["datetime"].max()
            - timeline["datetime"].min()
        ).total_seconds()
        / 3600,
        2
    )

    return pd.DataFrame([result])


import pandas as pd

# 전체 통계

def traffic_stats_all(datasets):

    rows = []

    for client_id, data in datasets.items():

        timeline = data["timeline"]

        start_time = timeline["datetime"].min()
        end_time = timeline["datetime"].max()

        rows.append({

            "client":
                client_id,

            "total_cars":
                timeline["car_id"].nunique(),

            "total_rows":
                len(timeline),

            "duration_hours":
                round(
                    (
                        end_time - start_time
                    ).total_seconds()
                    / 3600,
                    2
                ),

            "avg_rows_per_car":
                round(
                    timeline
                    .groupby("car_id")
                    .size()
                    .mean(),
                    2
                )
        })

    return pd.DataFrame(rows)



# 차량별 trajectory 길이

def traffic_distribution(timeline):

    return (
        timeline
        .groupby("car_id")
        .size()
        .describe()
    )