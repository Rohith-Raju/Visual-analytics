import pandas as pd
import plotly.graph_objects as go
import glob
from tqdm import tqdm
import numpy as np

# --- Configuration ---
DATA_DIR = "VAST-Challenge-2022/Datasets/"
LOG_FILES_PATTERN = f"{DATA_DIR}/Activity_Logs/ParticipantStatusLogs*.csv"
ATTRIBUTE_FILES = [
    f"{DATA_DIR}/Attributes/Buildings.csv",
    f"{DATA_DIR}/Attributes/Apartments.csv",
    f"{DATA_DIR}/Attributes/Employers.csv",
    f"{DATA_DIR}/Attributes/Pubs.csv",
    f"{DATA_DIR}/Attributes/Restaurants.csv",
    f"{DATA_DIR}/Attributes/Schools.csv",
]

# --- 1. Load Base Map Data ---
base_map_points_list = []
print("Creating base map from attribute files...")
for file_path in ATTRIBUTE_FILES:
    try:
        if not glob.glob(file_path):
            continue
        df_attr = pd.read_csv(file_path)
        if "location" in df_attr.columns:
            point_locs = df_attr[
                df_attr["location"].astype(str).str.startswith("POINT", na=False)
            ].copy()
            if not point_locs.empty:
                coords = (
                    point_locs["location"]
                    .astype(str)
                    .str.replace("POINT \\(", "", regex=True)
                    .str.replace("\\)", "", regex=True)
                    .str.split(" ", expand=True)
                )
                point_locs["x"] = pd.to_numeric(coords[0], errors="coerce")
                point_locs["y"] = pd.to_numeric(coords[1], errors="coerce")
                point_locs.dropna(subset=["x", "y"], inplace=True)
                if not point_locs.empty:
                    base_map_points_list.append(point_locs[["x", "y"]])
    except pd.errors.EmptyDataError:
        continue
    except Exception as e:
        print(f"Error processing attribute file {file_path}: {e}")

base_map_trace = None
if base_map_points_list:
    base_map_df = pd.concat(base_map_points_list, ignore_index=True)
    base_map_trace = go.Scattergl(
        x=base_map_df["x"],
        y=base_map_df["y"],
        mode="markers",
        marker=dict(size=2, color="lightgrey", opacity=0.5),
        name="City Locations",
        hoverinfo="none",
        showlegend=False,
    )
    print(f"Added {len(base_map_df)} base map points.")
else:
    print("No base map points generated.")

# --- 2. Process Traffic Data ---
traffic_points_list = []
log_files = glob.glob(LOG_FILES_PATTERN)
if not log_files:
    print(f"Error: No activity log files found: {LOG_FILES_PATTERN}")

print(f"\nProcessing {len(log_files)} activity log files for traffic data...")
for file_path in tqdm(log_files, desc="Processing Logs"):
    try:
        df_log = pd.read_csv(
            file_path, usecols=["timestamp", "currentLocation", "currentMode"]
        )
        df_transport = df_log[df_log["currentMode"] == "Transport"].copy()
        df_log = None

        if not df_transport.empty:
            coords = (
                df_transport["currentLocation"]
                .astype(str)
                .str.replace("POINT \\(", "", regex=True)
                .str.replace("\\)", "", regex=True)
                .str.split(" ", expand=True)
            )
            df_transport["x"] = pd.to_numeric(coords[0], errors="coerce")
            df_transport["y"] = pd.to_numeric(coords[1], errors="coerce")
            df_transport.dropna(subset=["x", "y"], inplace=True)

            if not df_transport.empty:
                df_transport["timestamp"] = pd.to_datetime(
                    df_transport["timestamp"], errors="coerce"
                )
                df_transport.dropna(subset=["timestamp"], inplace=True)
                if not df_transport.empty:
                    df_transport["hour_interval"] = (
                        df_transport["timestamp"].dt.hour // 3 * 3
                    )
                    df_transport["day_name"] = df_transport[
                        "timestamp"
                    ].dt.day_name()
                    traffic_points_list.append(
                        df_transport[["x", "y", "hour_interval", "day_name"]]
                    )
    except pd.errors.EmptyDataError:
        continue
    except Exception as e:
        print(f"Error processing log file {file_path}: {e}")

if not traffic_points_list:
    print("No traffic data processed.")
    traffic_df = pd.DataFrame(columns=["x", "y", "hour_interval", "day_name"])
else:
    traffic_df = pd.concat(traffic_points_list, ignore_index=True)
    traffic_points_list = None
    print(f"Total traffic points processed: {len(traffic_df)}")

# --- 3. Prepare Data and Traces for Heatmaps ---
all_plotly_traces = []
if base_map_trace:
    all_plotly_traces.append(base_map_trace)

heatmap_trace_metadata = []
unique_days = ["NoData"]
unique_intervals = [-1]
initial_title_text = "Traffic Density Heatmap (No Data)"

if not traffic_df.empty:
    unique_days = sorted(traffic_df["day_name"].unique().tolist())
    unique_intervals = sorted(traffic_df["hour_interval"].unique().tolist())

    if not unique_days: unique_days = ["NoDataDay"]
    if not unique_intervals: unique_intervals = [-1]

    x_min, x_max = traffic_df["x"].min(), traffic_df["x"].max()
    y_min, y_max = traffic_df["y"].min(), traffic_df["y"].max()
    n_bins = 100
    xbins = dict(
        start=x_min,
        end=x_max,
        size=(x_max - x_min) / n_bins if n_bins > 0 and x_max > x_min else 1,
    )
    ybins = dict(
        start=y_min,
        end=y_max,
        size=(y_max - y_min) / n_bins if n_bins > 0 and y_max > y_min else 1,
    )

    print(
        f"\nCreating {len(unique_days) * len(unique_intervals)} heatmap traces..."
    )
    for day_val in unique_days:
        for interval_val in unique_intervals:
            df_subset = traffic_df[
                (traffic_df["day_name"] == day_val)
                & (traffic_df["hour_interval"] == interval_val)
            ]
            is_initially_visible = (
                day_val == unique_days[0] and interval_val == unique_intervals[0]
            )
            current_trace = go.Histogram2d(
                x=df_subset["x"],
                y=df_subset["y"],
                xbins=xbins,
                ybins=ybins,
                colorscale="Hot",
                zsmooth="best",
                name=f"{day_val} {interval_val:02d}h",
                visible=is_initially_visible,
                showscale=is_initially_visible,
                hoverinfo="z",
                # Store day and interval for easy filtering in visibility updates
                customdata=np.array([[day_val, interval_val]] * len(df_subset)),
                meta={"day": day_val, "interval": interval_val} # For easier access
            )
            all_plotly_traces.append(current_trace)
            # Metadata for mapping (day, interval) to trace index in all_plotly_traces
            heatmap_trace_metadata.append(
                {
                    "day": day_val,
                    "interval": interval_val,
                    "trace_index": len(all_plotly_traces) - 1,
                }
            )
    if unique_days[0] != "NoData" and unique_intervals[0] != -1:
        initial_title_text = (
            f"Traffic: {unique_days[0]}, "
            f"{unique_intervals[0]:02d}:00-"
            f"{(unique_intervals[0] + 2):02d}:59"
        )

# --- 4. Create Controls ---
updatemenus_list = []
sliders_list = []

# Helper to generate visibility list and title object
def get_visibility_and_title_args(
    target_day, target_interval, all_traces_list, base_map_exists,
    heatmap_meta_list
):
    visibility = [False] * len(all_traces_list)
    if base_map_exists:
        visibility[0] = True

    new_title_str = f"Traffic: {target_day}, {target_interval:02d}:00-{(target_interval + 2):02d}:59"
    
    active_heatmap_for_scale_idx = -1

    for meta in heatmap_meta_list:
        trace = all_traces_list[meta["trace_index"]]
        is_target = (meta["day"] == target_day and meta["interval"] == target_interval)
        visibility[meta["trace_index"]] = is_target
        if is_target and active_heatmap_for_scale_idx == -1:
            active_heatmap_for_scale_idx = meta["trace_index"]

    # Prepare trace updates for showscale (complex to do in one go for all traces)
    # Simpler: rely on initial setup and ensure only one is true in visibility
    # Forcing showscale via layout update is more robust if needed
    # For now, the visibility array itself handles which trace is shown.
    # The showscale property was set at trace creation.

    return {"visible": visibility}, {"title.text": new_title_str}


# Day Dropdown
day_buttons_list = []
if unique_days[0] != "NoData" and unique_intervals[0] != -1:
    for day_idx, current_day_name in enumerate(unique_days):
        # Action for this day button:
        # 1. Set view to (current_day_name, first_interval)
        # 2. Reset slider to first step
        # 3. Reprogram ALL slider steps to use current_day_name

        first_interval_val = unique_intervals[0]
        
        # Args for the immediate update when this day button is clicked
        vis_args_for_day_button, title_args_for_day_button = get_visibility_and_title_args(
            current_day_name, first_interval_val, all_plotly_traces,
            base_map_trace is not None, heatmap_trace_metadata
        )

        # Prepare layout updates, including reprogramming slider steps
        layout_updates_for_day_button = {
            "title.text": title_args_for_day_button["title.text"],
            "sliders[0].active": 0,  # Reset slider to first step
        }

        # Reprogram each slider step's args
        for interval_s_idx, interval_s_val in enumerate(unique_intervals):
            vis_args_slider, title_args_slider = get_visibility_and_title_args(
                current_day_name, # THIS DAY
                interval_s_val,   # Slider's interval
                all_plotly_traces,
                base_map_trace is not None,
                heatmap_trace_metadata
            )
            # Path to update the specific slider step's args
            layout_updates_for_day_button[f"sliders[0].steps[{interval_s_idx}].args"] = [
                vis_args_slider, title_args_slider
            ]
        
        day_buttons_list.append(
            dict(
                label=current_day_name,
                method="update",
                args=[
                    vis_args_for_day_button, # Update data visibility
                    layout_updates_for_day_button # Update layout (title, slider active, slider steps)
                ],
            )
        )
    if day_buttons_list:
        updatemenus_list.append(
            dict(
                type="dropdown", direction="down", x=0.01, y=1.12, showactive=True,
                buttons=day_buttons_list, xanchor="left", yanchor="top", active=0,
                pad={"t":5, "b":5}
            )
        )

# Interval Slider
slider_steps_list = []
if unique_intervals[0] != -1 and unique_days[0] != "NoData":
    # Initial definition of slider steps. These will be reprogrammed by the day dropdown.
    # For the initial state (before any dropdown click), they operate on unique_days[0].
    initial_day_for_slider = unique_days[0]
    for interval_idx, interval_val in enumerate(unique_intervals):
        vis_arg_slider_step, title_arg_slider_step = get_visibility_and_title_args(
            initial_day_for_slider, # Default to first day
            interval_val,
            all_plotly_traces,
            base_map_trace is not None,
            heatmap_trace_metadata
        )
        slider_steps_list.append(
            dict(
                label=f"{interval_val:02d}-{(interval_val + 2):02d}h",
                method="update",
                args=[vis_arg_slider_step, title_arg_slider_step],
            )
        )
    if slider_steps_list:
        sliders_list.append(
            dict(
                active=0, # Corresponds to the first interval
                currentvalue={"prefix": "Time: ", "font": {"size": 14}},
                pad={"t": 10, "b":10},
                steps=slider_steps_list,
                x=0.5, xanchor="center", y=0.02, yanchor="top", len=0.9, lenmode='fraction'
            )
        )

# --- 5. Create and Show Figure ---
fig = go.Figure(data=all_plotly_traces)

fig.update_layout(
    title_text=initial_title_text,
    title_x=0.5,
    xaxis_title="X Coordinate",
    yaxis_title="Y Coordinate",
    yaxis=dict(scaleanchor="x", scaleratio=1, autorange=True), # Ensure autorange for y if x changes
    xaxis=dict(autorange=True),
    plot_bgcolor="white",
    margin=dict(l=40, r=40, t=80, b=80), # Adjusted top margin for dropdown
    updatemenus=updatemenus_list,
    sliders=sliders_list,
    legend=dict(traceorder="reversed", title_text="Layers", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

# Apply the reprogramming for the initially active day in the dropdown (day 0)
# This ensures the slider is correctly programmed on load for the first day.
if day_buttons_list and 'args' in day_buttons_list[0] and len(day_buttons_list[0]['args']) > 1:
    initial_layout_updates = day_buttons_list[0]['args'][1]
    fig.update_layout(initial_layout_updates)


print("\nInteraction Note: Select a day from the dropdown. This will set the day context and also reprogram the interval slider to operate within that selected day.")
fig.show()
