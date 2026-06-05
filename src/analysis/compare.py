import pandas as pd


import pandas as pd


def compare_clients(datasets):
    """
    DT1~DT3 규모 비교
    """

    rows = []

    for client_id, data in datasets.items():

        timeline = data["timeline"]
        final = data["final"]

        start_time = timeline["datetime"].min()
        end_time = timeline["datetime"].max()

        duration_hours = round(
            (end_time - start_time).total_seconds() / 3600,
            2
        )

        rows.append({
            "Client": client_id,
            "Total Cars": timeline["car_id"].nunique(),
            "Parking Cars": len(final),
            "Non Parking Cars": (
                timeline["car_id"].nunique()
                - len(final)
            ),
            "Slots": (
                timeline["park_slot_id"]
                .dropna()
                .nunique()
            ),
            "Timeline Rows": len(timeline),
            "Duration(H)": duration_hours
        })

    return pd.DataFrame(rows)



def compare_duration(datasets):
    """
    체류시간 비교
    """

    rows = []

    for client_id, data in datasets.items():

        final = data["final"]

        if "duration" not in final.columns:
            continue

        rows.append({

            "Client": client_id,

            "Avg Duration":
                round(
                    final["duration"].mean(),
                    2
                ),

            "Median Duration":
                round(
                    final["duration"].median(),
                    2
                ),

            "Max Duration":
                round(
                    final["duration"].max(),
                    2
                ),

            "Min Duration":
                round(
                    final["duration"].min(),
                    2
                )
        })

    return pd.DataFrame(rows)


def compare_demand(datasets):
    """
    수요(Arrival Count) 비교
    """

    rows = []

    for client_id, data in datasets.items():

        timeline = data["timeline"]

        first_seen = (
            timeline
            .groupby("car_id")["datetime"]
            .min()
            .reset_index()
        )

        arrival = (
            first_seen
            .set_index("datetime")
            .resample("5min")
            .size()
        )

        rows.append({

            "Client":
                client_id,

            "Avg Arrival":
                round(
                    arrival.mean(),
                    2
                ),

            "Peak Arrival":
                int(
                    arrival.max()
                ),

            "Peak Time":
                arrival.idxmax()
        })

    return pd.DataFrame(rows)


def compare_slots(datasets):
    """
    슬롯 사용 비교
    """

    rows = []

    for client_id, data in datasets.items():

        timeline = data["timeline"]

        slot_usage = (
            timeline["park_slot_id"]
            .dropna()
            .value_counts()
        )

        rows.append({

            "Client":
                client_id,

            "Slot Count":
                slot_usage.shape[0],

            "Most Used Slot":
                slot_usage.idxmax(),

            "Usage Count":
                int(
                    slot_usage.max()
                )
        })

    return pd.DataFrame(rows)