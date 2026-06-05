import pandas as pd


def duration_stats(final):

    duration_col = "duration"

    return (
        final[duration_col]
        .describe()
        .to_frame()
    )


# 평균 체류시간

def avg_duration(final):

    return final["duration"].mean()

# 슬롯별 체류시간

def duration_by_slot(final):

    return (
        final
        .groupby("park_slot_id")
        ["duration"]
        .mean()
        .sort_values(
            ascending=False
        )
    )

import pandas as pd

# 전체 비교

def duration_stats_all(datasets):

    rows = []

    for client_id, data in datasets.items():

        final = data["final"]

        rows.append({

            "client":
                client_id,

            "avg_duration":
                round(
                    final["duration"]
                    .mean(),
                    2
                ),

            "median_duration":
                round(
                    final["duration"]
                    .median(),
                    2
                ),

            "max_duration":
                round(
                    final["duration"]
                    .max(),
                    2
                ),

            "min_duration":
                round(
                    final["duration"]
                    .min(),
                    2
                )
        })

    return pd.DataFrame(rows)