import pandas as pd


def parking_stats(timeline):

    slots = (
        timeline["park_slot_id"]
        .dropna()
    )

    result = {}

    result["slot_count"] = slots.nunique()

    result["parking_events"] = len(slots)

    result["occupancy_ratio"] = (
        len(slots)
        / len(timeline)
    )

    return pd.DataFrame([result])

# 슬롯 사용량

def slot_usage(timeline):

    return (
        timeline["park_slot_id"]
        .value_counts()
        .sort_values(
            ascending=False
        )
    )

# 슬롯 점유율

def slot_occupancy_rate(timeline):

    slot_counts = (
        timeline["park_slot_id"]
        .value_counts()
    )

    return (
        slot_counts
        / slot_counts.sum()
    )


import pandas as pd


# 전체 통계

def parking_stats_all(datasets):

    rows = []

    for client_id, data in datasets.items():

        timeline = data["timeline"]

        slots = (
            timeline["park_slot_id"]
            .dropna()
        )

        rows.append({

            "client":
                client_id,

            "slot_count":
                slots.nunique(),

            "parking_events":
                len(slots),

            "occupancy_ratio":
                round(
                    len(slots)
                    / len(timeline),
                    4
                )
        })

    return pd.DataFrame(rows)