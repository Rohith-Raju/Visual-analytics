import pandas as pd
from datetime import datetime, time
import glob # For finding multiple files
import plotly.express as px
import plotly.graph_objects as go

# --- Configuration ---
DATA_DIR = "VAST-Challenge-2022/Datasets/"
LOG_FILES_PATTERN = f"{DATA_DIR}/Activity_Logs/ParticipantStatusLogs*.csv"
PARTICIPANTS_FILE = f"{DATA_DIR}/Attributes/Participants.csv"
TRAVEL_JOURNAL_FILE = f"{DATA_DIR}/Journals/TravelJournal.csv"
FINANCIAL_JOURNAL_FILE = f"{DATA_DIR}/Journals/FinancialJournal.csv"
CHECKIN_JOURNAL_FILE = f"{DATA_DIR}/Journals/CheckinJournal.csv"

SELECTED_PARTICIPANT_IDS = [4, 171] # Initial selection, likely needs changing
TARGET_DATE_STR = "2022-03-01"        # Initial target date, likely needs changing
TARGET_DATE = datetime.strptime(TARGET_DATE_STR, "%Y-%m-%d").date()
NUM_LOG_FILES_TO_LOAD = 5 # Set to a small number for testing, or None to load all

# --- Plotting Configuration ---
MODE_COLORS = {
    "AtHome": "blue",
    "Transport": "lightgrey", # Will be mostly overlaid by specific travel tasks
    "AtWork": "green",
    "AtRestaurant": "orange",
    "AtRecreation": "purple",
    "School": "cyan", # Assuming 'School' might be a mode
    "Unknown": "black"
}
TRAVEL_COLOR = "teal" # Specific color for travel segments from TravelJournal
FINANCIAL_MARKER_COLOR_EXPENSE = "red"
FINANCIAL_MARKER_COLOR_INCOME = "limegreen"

def load_and_preprocess_data():
    """Loads and preprocesses all necessary dataframes."""
    data = {'logs': pd.DataFrame(), 'participants': pd.DataFrame(), 'travel': pd.DataFrame(), 'financial': pd.DataFrame(), 'checkin': pd.DataFrame()}
    all_log_files = sorted(glob.glob(LOG_FILES_PATTERN))

    if not all_log_files:
        print(f"Error: No log files found matching pattern {LOG_FILES_PATTERN}")
        return data

    files_to_process = all_log_files
    if NUM_LOG_FILES_TO_LOAD is not None and NUM_LOG_FILES_TO_LOAD > 0 and NUM_LOG_FILES_TO_LOAD < len(all_log_files):
        files_to_process = all_log_files[:NUM_LOG_FILES_TO_LOAD]
        print(f"Found {len(all_log_files)} log files. Processing the first {len(files_to_process)}: {files_to_process}")
    else:
        print(f"Found {len(all_log_files)} log files. Loading all of them ({len(files_to_process)} files)...")

    df_list = []
    for filename in files_to_process:
        try:
            df_temp = pd.read_csv(filename)
            df_list.append(df_temp)
        except pd.errors.EmptyDataError:
            print(f"  Warning: {filename} is empty. Skipping.")
        except Exception as e:
            print(f"  Error loading {filename}: {e}. Skipping.")

    if df_list:
        data['logs'] = pd.concat(df_list, ignore_index=True)
        data['logs']['timestamp'] = pd.to_datetime(data['logs']['timestamp'])
        print(f"  Log files loaded. Total shape: {data['logs'].shape}")
        if not data['logs'].empty:
            min_log_date = data['logs']['timestamp'].min().date()
            max_log_date = data['logs']['timestamp'].max().date()
            print(f"  Loaded logs date range: {min_log_date} to {max_log_date}")
            if not (min_log_date <= TARGET_DATE <= max_log_date):
                print(f"  WARNING: TARGET_DATE {TARGET_DATE_STR} is outside loaded log range.")
    else:
        print("No log data loaded.")

    try:
        print(f"Loading {PARTICIPANTS_FILE}...")
        data['participants'] = pd.read_csv(PARTICIPANTS_FILE)
        print(f"Loading {TRAVEL_JOURNAL_FILE}...")
        data['travel'] = pd.read_csv(TRAVEL_JOURNAL_FILE)
        data['travel']['travelStartTime'] = pd.to_datetime(data['travel']['travelStartTime'])
        data['travel']['travelEndTime'] = pd.to_datetime(data['travel']['travelEndTime'])
        # Ensure consistent timezone handling or make naive for comparison if appropriate
        if data['travel']['travelStartTime'].dt.tz is not None:
             print(f"  TravelJournal start times are tz-aware: {data['travel']['travelStartTime'].dt.tz}")
        if data['travel']['travelEndTime'].dt.tz is not None:
             print(f"  TravelJournal end times are tz-aware: {data['travel']['travelEndTime'].dt.tz}")

        print(f"Loading {FINANCIAL_JOURNAL_FILE}...")
        data['financial'] = pd.read_csv(FINANCIAL_JOURNAL_FILE)
        data['financial']['timestamp'] = pd.to_datetime(data['financial']['timestamp'])
        print(f"Loading {CHECKIN_JOURNAL_FILE}...")
        data['checkin'] = pd.read_csv(CHECKIN_JOURNAL_FILE)
        data['checkin']['timestamp'] = pd.to_datetime(data['checkin']['timestamp'])
        print("Attribute and Journal data loading complete.")
    except FileNotFoundError as e:
        print(f"Error: File not found for attributes/journals. {e}")
    except Exception as e:
        print(f"Error loading attributes/journals: {e}")
    return data

def describe_and_prepare_plot_data(participant_id, target_date, all_data):
    """Describes daily pattern and prepares data for plotting."""
    print(f"\n--- Daily Pattern for Participant ID: {participant_id} on {target_date.strftime('%Y-%m-%d')} ---")

    timeline_tasks = []
    financial_markers = []

    if all_data['participants'].empty:
        print("Participant attribute data (Participants.csv) not available or empty.")
    else:
        participant_info_df = all_data['participants'][all_data['participants']['participantId'] == participant_id]
        if participant_info_df.empty:
            print(f"No static information found for participant {participant_id} in Participants.csv.")
        else:
            participant_info = participant_info_df.iloc[0]
            print(f"Participant Info: Age {participant_info.get('age', 'N/A')}, Education {participant_info.get('educationLevel', 'N/A')}, Job ID: {participant_info.get('jobId', 'N/A')}")

    if all_data['logs'].empty:
        print("No activity log data loaded to analyze.")
        return None, None

    p_logs = all_data['logs'][
        (all_data['logs']['participantId'] == participant_id) &
        (all_data['logs']['timestamp'].dt.date == target_date)
    ].sort_values(by='timestamp').reset_index(drop=True)

    if p_logs.empty:
        print("No activity logs found for this participant on this specific date.")
        return None, None

    # --- Prepare Mode Segments for Plotting ---
    current_segment_start_time = None
    current_segment_mode = None
    log_tz = p_logs['timestamp'].dt.tz # Get timezone from logs if available

    for i, log_row in p_logs.iterrows():
        ts = log_row['timestamp']
        mode = log_row.get('currentMode', 'Unknown')

        if current_segment_mode is None:
            current_segment_start_time = ts
            current_segment_mode = mode
        elif mode != current_segment_mode:
            timeline_tasks.append(dict(
                Task=f"{current_segment_mode}", # Simpler task name
                Start=current_segment_start_time,
                Finish=ts,
                Resource=current_segment_mode,
                Participant=str(participant_id),
                Type="Mode"
            ))
            current_segment_start_time = ts
            current_segment_mode = mode

        if i == len(p_logs) - 1: # Last log entry
            segment_end_time = ts + pd.Timedelta(minutes=4, seconds=59) # End of this 5-min interval
            day_end_dt = datetime.combine(target_date, time(23, 59, 59, 999999))
            day_end_ts = pd.Timestamp(day_end_dt)
            if log_tz: day_end_ts = day_end_ts.tz_localize(log_tz)

            if segment_end_time > day_end_ts : segment_end_time = day_end_ts

            timeline_tasks.append(dict(
                Task=f"{current_segment_mode}",
                Start=current_segment_start_time,
                Finish=segment_end_time,
                Resource=current_segment_mode,
                Participant=str(participant_id),
                Type="Mode"
            ))

    # --- Prepare Travel Data for Plotting ---
    if not all_data['travel'].empty:
        p_travel = all_data['travel'][
            (all_data['travel']['participantId'] == participant_id)
        ].copy()

        start_of_target_day = pd.Timestamp(datetime.combine(target_date, time.min))
        end_of_target_day = pd.Timestamp(datetime.combine(target_date, time.max))

        travel_tz = p_travel['travelStartTime'].dt.tz
        if travel_tz: # If travel times are tz-aware
            start_of_target_day = start_of_target_day.tz_localize(travel_tz)
            end_of_target_day = end_of_target_day.tz_localize(travel_tz)
        # If travel times are naive, and log_tz is available, consider converting travel times
        elif log_tz and p_travel['travelStartTime'].dt.tz is None:
            print("Warning: Travel times are tz-naive but log times are tz-aware. Assuming travel times are in the same timezone as logs for plotting.")
            p_travel['travelStartTime'] = p_travel['travelStartTime'].apply(lambda x: pd.Timestamp(x).tz_localize(log_tz) if pd.notnull(x) else x)
            p_travel['travelEndTime'] = p_travel['travelEndTime'].apply(lambda x: pd.Timestamp(x).tz_localize(log_tz) if pd.notnull(x) else x)
            start_of_target_day = start_of_target_day.tz_localize(log_tz) # Now make these aware too
            end_of_target_day = end_of_target_day.tz_localize(log_tz)


        # Filter for travel overlapping the target day
        p_travel_on_day = p_travel[
            (p_travel['travelStartTime'] <= end_of_target_day) &
            (p_travel['travelEndTime'] >= start_of_target_day)
        ]

        for _, tr_row in p_travel_on_day.iterrows():
            plot_start = max(tr_row['travelStartTime'], start_of_target_day)
            plot_finish = min(tr_row['travelEndTime'], end_of_target_day)
            if plot_start < plot_finish:
                timeline_tasks.append(dict(
                    Task=f"Travel: {tr_row.get('purpose', 'N/A')}",
                    Start=plot_start,
                    Finish=plot_finish,
                    Resource="Travel",
                    Participant=str(participant_id),
                    Type="Travel"
                ))

    # --- Prepare Financial Data for Plotting ---
    if not all_data['financial'].empty:
        p_financial = all_data['financial'][
            (all_data['financial']['participantId'] == participant_id) &
            (all_data['financial']['timestamp'].dt.date == target_date)
        ]
        for _, fin_row in p_financial.iterrows():
            financial_markers.append(dict(
                Timestamp=fin_row['timestamp'],
                Amount=fin_row.get('amount', 0),
                Category=fin_row.get('category', 'N/A'),
                Participant=str(participant_id)
            ))
    # --- (Textual description part can be added here if desired) ---
    # For brevity, focusing on plot data preparation.
    # The original textual print loop from `describe_participant_day` can be re-inserted here.
    print("Plot data prepared.")
    return timeline_tasks, financial_markers


def plot_participant_routine(participant_id, target_date, timeline_tasks, financial_markers):
    """Plots the participant's daily routine using Plotly."""
    if not timeline_tasks and not financial_markers:
        print(f"No data to plot for participant {participant_id} on {target_date.strftime('%Y-%m-%d')}.")
        return

    fig_title = f"Daily Routine for Participant {participant_id} on {target_date.strftime('%Y-%m-%d')}"
    
    # Create color map for timeline tasks
    color_map = MODE_COLORS.copy()
    color_map["Travel"] = TRAVEL_COLOR # Add specific color for travel resource

    if timeline_tasks:
        df_timeline = pd.DataFrame(timeline_tasks)
        # Ensure Start and Finish are datetime objects
        df_timeline['Start'] = pd.to_datetime(df_timeline['Start'])
        df_timeline['Finish'] = pd.to_datetime(df_timeline['Finish'])

        # Create the main timeline figure
        fig = px.timeline(
            df_timeline,
            x_start="Start",
            x_end="Finish",
            y="Participant", # Using Participant ID on Y-axis
            color="Resource", # Color by Mode or "Travel"
            color_discrete_map=color_map,
            hover_name="Task",
            title=fig_title
        )
        fig.update_yaxes(categoryorder="array", categoryarray=[str(participant_id)]) # Ensure single participant is shown correctly
    else: # If only financial markers
        fig = go.Figure()
        fig.update_layout(title=fig_title,
                          xaxis_title="Time",
                          yaxis_title="Participant",
                          yaxis=dict(categoryorder="array", categoryarray=[str(participant_id)], showticklabels=True, title_text=str(participant_id)),
                          # Set x-axis range for the whole day
                          xaxis_range=[datetime.combine(target_date, time.min), datetime.combine(target_date, time.max)])


    # Add financial transactions as scatter markers
    if financial_markers:
        df_financial = pd.DataFrame(financial_markers)
        df_financial['Timestamp'] = pd.to_datetime(df_financial['Timestamp'])
        
        for _, row in df_financial.iterrows():
            marker_color = FINANCIAL_MARKER_COLOR_EXPENSE if row['Amount'] < 0 else FINANCIAL_MARKER_COLOR_INCOME
            marker_symbol = "triangle-down" if row['Amount'] < 0 else "triangle-up"
            fig.add_trace(go.Scatter(
                x=[row['Timestamp']],
                y=[str(row['Participant'])], # Plot on the same y-level as timeline
                mode='markers',
                marker=dict(color=marker_color, size=10, symbol=marker_symbol),
                name=f"Financial: {row['Category']} (${row['Amount']:.2f})",
                hovertext=f"{row['Category']}: ${row['Amount']:.2f}<br>Time: {row['Timestamp'].strftime('%H:%M')}",
                hoverinfo="text"
            ))

    fig.update_layout(
        xaxis_title="Time of Day",
        yaxis_title="Activity Type / Participant", # Y-axis is participant ID
        showlegend=True,
        legend_title_text='Legend'
    )
    # Ensure x-axis covers the whole day
    x_axis_start = pd.Timestamp(datetime.combine(target_date, time.min))
    x_axis_end = pd.Timestamp(datetime.combine(target_date, time.max))
    
    # If data is tz-aware, make axis range tz-aware
    if timeline_tasks and pd.api.types.is_datetime64_any_dtype(df_timeline['Start']) and df_timeline['Start'].dt.tz is not None:
        tzinfo = df_timeline['Start'].dt.tz
        x_axis_start = x_axis_start.tz_localize(tzinfo)
        x_axis_end = x_axis_end.tz_localize(tzinfo)
    elif financial_markers and pd.api.types.is_datetime64_any_dtype(df_financial['Timestamp']) and df_financial['Timestamp'].dt.tz is not None:
        tzinfo = df_financial['Timestamp'].dt.tz
        x_axis_start = x_axis_start.tz_localize(tzinfo)
        x_axis_end = x_axis_end.tz_localize(tzinfo)

    fig.update_xaxes(range=[x_axis_start, x_axis_end])

    fig.show()


if __name__ == "__main__":
    all_data = load_and_preprocess_data()

    if all_data and not all_data['logs'].empty:
        # --- Helper to find active participants on a given date ---
        # (Keep the helper commented out or use it to find good PIDs/Dates)
        '''
        test_date_str = "2022-01-03" # <<< CHANGE THIS to a date within your loaded log range
        try:
            test_date = datetime.strptime(test_date_str, "%Y-%m-%d").date()
            if not all_data['logs'].empty:
                active_p_on_test_date = all_data['logs'][all_data['logs']['timestamp'].dt.date == test_date]['participantId'].unique()
                if len(active_p_on_test_date) > 0:
                    print(f"\n--- HELPER: Participants active on {test_date_str} ---")
                    print(f"Found {len(active_p_on_test_date)} active participants. First 10 (or fewer): {active_p_on_test_date[:10].tolist()}")
                    print(f"Consider updating SELECTED_PARTICIPANT_IDS with two of these and TARGET_DATE_STR to {test_date_str}")
                else:
                    print(f"\n--- HELPER: No participants found active on {test_date_str} in the loaded logs.")
            else:
                print("\n--- HELPER: No log data loaded to search for active participants.")
        except ValueError:
            print(f"\n--- HELPER: Invalid date format for test_date_str: '{test_date_str}'. Please use YYYY-MM-DD.")
        '''
        # --- End Helper ---

        print(f"\nAnalyzing pre-selected participants ({SELECTED_PARTICIPANT_IDS}) for date: {TARGET_DATE_STR}")
        valid_participant_ids_to_analyze = []

        if all_data['participants'].empty:
            print("Warning: Participants.csv is empty or not loaded. Will attempt to analyze IDs if they have logs for the target date.")
            for pid in SELECTED_PARTICIPANT_IDS:
                if not all_data['logs'][(all_data['logs']['participantId'] == pid) & (all_data['logs']['timestamp'].dt.date == TARGET_DATE)].empty:
                    valid_participant_ids_to_analyze.append(pid)
                else:
                    print(f"Note: Participant ID {pid} (initial selection) has no logs for {TARGET_DATE_STR} in loaded files.")
        else:
            for pid in SELECTED_PARTICIPANT_IDS:
                if pid in all_data['participants']['participantId'].values:
                    if not all_data['logs'][(all_data['logs']['participantId'] == pid) & (all_data['logs']['timestamp'].dt.date == TARGET_DATE)].empty:
                        valid_participant_ids_to_analyze.append(pid)
                    else:
                        print(f"Note: Participant ID {pid} (in Participants.csv) has no logs for {TARGET_DATE_STR} in loaded files.")
                else:
                    print(f"Warning: Predefined Participant ID {pid} not found in Participants.csv.")

        if not valid_participant_ids_to_analyze:
            print(f"\nNo participants from your current selection ({SELECTED_PARTICIPANT_IDS}) have activity logs for {TARGET_DATE_STR} in the loaded data.")
            print("Please use the HELPER section to find suitable Participant IDs and a TARGET_DATE_STR where they have data.")
        else:
            print(f"Proceeding with analysis for participants: {valid_participant_ids_to_analyze} on {TARGET_DATE_STR}")
            for p_id in valid_participant_ids_to_analyze:
                timeline_data, financial_data = describe_and_prepare_plot_data(p_id, TARGET_DATE, all_data)
                if timeline_data is not None or financial_data is not None: # Check if any data was prepared
                    plot_participant_routine(p_id, TARGET_DATE, timeline_data, financial_data)
    else:
        print("Could not load sufficient log data to proceed. Exiting.")
