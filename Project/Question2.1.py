import pandas as pd
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# --- 1. Load and preprocess data ---
tj = pd.read_csv("VAST-Challenge-2022/Datasets/Journals/TravelJournal.csv")
al = pd.read_csv("VAST-Challenge-2022/Datasets/Activity_Logs/ParticipantStatusLogs1.csv")

tj["travelStartTime"] = pd.to_datetime(tj["travelStartTime"])
tj["travelEndTime"] = pd.to_datetime(tj["travelEndTime"])
al["timestamp"] = pd.to_datetime(al["timestamp"])

# Extract x, y from WKT POINT
al[["x", "y"]] = (
    al["currentLocation"]
    .str.replace("POINT \(", "", regex=True)
    .str.replace("\)", "", regex=True)
    .str.split(" ", expand=True)
    .astype(float)
)

al = al[al["currentMode"] == "Transport"]
al["day_name"] = al["timestamp"].dt.day_name()
tj["day_name"] = tj["travelStartTime"].dt.day_name()

days_of_interest = ["Tuesday", "Saturday"]
tj = tj[tj["day_name"].isin(days_of_interest)]
al = al[al["day_name"].isin(days_of_interest)]

# Focus on top 3 purposes
purposes_of_interest = tj["purpose"].value_counts().index[:3]
tj = tj[tj["purpose"].isin(purposes_of_interest)]

def day_night(ts):
    hour = ts.hour
    return "Day" if 6 <= hour < 18 else "Night"

# --- 2. Pre-group activity logs by participantId for fast lookup ---
al_groups = {pid: group for pid, group in al.groupby("participantId")}

# --- 3. Parallel aggregation of trajectories ---
def process_trip(trip_row):
    trip = trip_row[1]
    pid = trip["participantId"]
    if pid not in al_groups:
        return None
    al_pid = al_groups[pid]
    mask = (
        (al_pid["timestamp"] >= trip["travelStartTime"]) &
        (al_pid["timestamp"] <= trip["travelEndTime"])
    )
    trip_points = al_pid.loc[mask, ["participantId", "timestamp", "x", "y", "day_name"]].copy()
    if trip_points.empty:
        return None
    trip_points["purpose"] = trip["purpose"]
    trip_points["trip_id"] = trip_row[0]
    trip_points["time_of_day"] = trip_points["timestamp"].apply(day_night)
    return trip_points

agg_rows = []
with ThreadPoolExecutor() as executor:
    results = list(tqdm(
        executor.map(process_trip, tj.iterrows()),
        total=len(tj)
    ))
agg_rows = [r for r in results if r is not None]
agg_df = pd.concat(agg_rows, ignore_index=True)

purposes = list(agg_df["purpose"].unique())
time_of_days = ["Day", "Night"]
days = days_of_interest
colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray"]
purpose_color = {p: colors[i % len(colors)] for i, p in enumerate(purposes)}

# --- 4. Build all traces (one for each (day, purpose, time_of_day, trip)) ---
traces = []
trace_labels = []  # (day, purpose, time_of_day)
for day in days:
    for time_of_day in time_of_days:
        for purpose in purposes:
            for trip_id, group in agg_df[
                (agg_df["purpose"] == purpose) &
                (agg_df["time_of_day"] == time_of_day) &
                (agg_df["day_name"] == day)
            ].groupby("trip_id"):
                group = group.sort_values("timestamp")
                traces.append(go.Scattergl(
                    x=group["x"],
                    y=group["y"],
                    mode="lines",
                    line=dict(width=1, color=purpose_color[purpose]),
                    opacity=0.15,
                    name=f"{purpose} ({time_of_day}, {day})",
                    legendgroup=f"{purpose}{time_of_day}{day}",
                    showlegend=False,
                    visible=(day == "Tuesday" and time_of_day == "Day"),  # Show Tuesday Day by default
                ))
                trace_labels.append((day, purpose, time_of_day))

for purpose in purposes:
    traces.append(go.Scattergl(
        x=[None], y=[None], mode="lines",
        line=dict(width=3, color=purpose_color[purpose]),
        name=purpose,
        legendgroup=purpose,
        showlegend=True,
        visible=True  
    ))
    trace_labels.append(("legend", purpose, "legend"))

n_legend = len(purposes)
total_traces = len(trace_labels)

def make_visibility_mask(selected_days, selected_purposes, selected_times):
    mask = []
    for (day, purpose, time_of_day) in trace_labels:
        if day == "legend":
            mask.append(True)  
        else:
            mask.append(
                (day in selected_days) and
                (purpose in selected_purposes) and
                (time_of_day in selected_times)
            )
    return mask

# 1. Day selector
day_buttons = []
for day in days:
    mask = make_visibility_mask([day], purposes, time_of_days)
    day_buttons.append(dict(
        label=day,
        method="update",
        args=[{"visible": mask}]
    ))
day_buttons.insert(0, dict(
    label="Both Days",
    method="update",
    args=[{"visible": make_visibility_mask(days, purposes, time_of_days)}]
))

# 2. Purpose selector
purpose_buttons = []
for purpose in purposes:
    mask = make_visibility_mask(days, [purpose], time_of_days)
    purpose_buttons.append(dict(
        label=purpose,
        method="update",
        args=[{"visible": mask}]
    ))
purpose_buttons.insert(0, dict(
    label="All Purposes",
    method="update",
    args=[{"visible": make_visibility_mask(days, purposes, time_of_days)}]
))

# 3. Day/Night selector
daynight_buttons = [
    dict(
        label="Day Only",
        method="update",
        args=[{"visible": make_visibility_mask(days, purposes, ["Day"])}]
    ),
    dict(
        label="Night Only",
        method="update",
        args=[{"visible": make_visibility_mask(days, purposes, ["Night"])}]
    ),
    dict(
        label="Both",
        method="update",
        args=[{"visible": make_visibility_mask(days, purposes, time_of_days)}]
    ),
]

# --- 6. Build the figure ---
fig = go.Figure(traces)
fig.update_layout(
    title="Actual Trajectories by Purpose, Time of Day, and Day",
    xaxis_title="X",
    yaxis_title="Y",
    yaxis=dict(scaleanchor="x", scaleratio=1),
    plot_bgcolor="white",
    margin=dict(l=20, r=20, t=40, b=20),
    updatemenus=[
        dict(
            type="dropdown",
            direction="down",
            x=0.01, y=1.15,
            showactive=True,
            buttons=day_buttons,
            xanchor="left",
            yanchor="top"
        ),
        dict(
            type="dropdown",
            direction="down",
            x=0.18, y=1.15,
            showactive=True,
            buttons=purpose_buttons,
            xanchor="left",
            yanchor="top"
        ),
        dict(
            type="buttons",
            direction="right",
            x=0.35, y=1.15,
            showactive=True,
            buttons=daynight_buttons,
            xanchor="left",
            yanchor="top"
        )
    ]
)
fig.show()
