import pandas as pd
from datetime import datetime, time
import glob
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import re

# --- Configuration ---
DATA_DIR = "VAST-Challenge-2022/Datasets/"
LOG_FILES_PATTERN = f"{DATA_DIR}/Activity_Logs/ParticipantStatusLogs*.csv"
TRAVEL_JOURNAL_FILE = f"{DATA_DIR}/Journals/TravelJournal.csv"
FINANCIAL_JOURNAL_FILE = f"{DATA_DIR}/Journals/FinancialJournal.csv"
PARTICIPANTS_FILE = f"{DATA_DIR}/Attributes/Participants.csv"

NUM_FILES_PER_PERIOD = 20
TOP_N_TRAVEL_PURPOSES = 7 # Number of top travel purposes to plot
PURPOSE_TO_EXCLUDE = "Going Back to Home" # Define the purpose to exclude

def natsort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'([0-9]+)', s)]

def load_selected_logs_and_journals():
    """Loads a subset of log files and relevant journal files."""
    data = {'early_logs': pd.DataFrame(), 'late_logs': pd.DataFrame(),
            'travel': pd.DataFrame(), 'financial': pd.DataFrame(),
            'participants': pd.DataFrame()}
    
    all_log_files_unsorted = glob.glob(LOG_FILES_PATTERN)
    if not all_log_files_unsorted:
        print(f"Error: No log files found matching pattern {LOG_FILES_PATTERN}")
        return None
    all_log_files = sorted(all_log_files_unsorted, key=natsort_key)
    
    if len(all_log_files) < NUM_FILES_PER_PERIOD * 2:
        print(f"Error: Not enough log files ({len(all_log_files)}) for {NUM_FILES_PER_PERIOD} files per period.")
        return None

    early_files = all_log_files[:NUM_FILES_PER_PERIOD]
    late_files = all_log_files[-NUM_FILES_PER_PERIOD:]

    print(f"Loading EARLY period logs ({len(early_files)} files): {early_files[0]}...{early_files[-1]}")
    df_list_early = []
    for filename in early_files:
        try: df_list_early.append(pd.read_csv(filename))
        except Exception as e: print(f"  Error loading {filename}: {e}")
    if df_list_early:
        data['early_logs'] = pd.concat(df_list_early, ignore_index=True)
        data['early_logs']['timestamp'] = pd.to_datetime(data['early_logs']['timestamp'])
        if not data['early_logs'].empty:
            print(f"  Early logs loaded: {data['early_logs']['timestamp'].min()} to {data['early_logs']['timestamp'].max()} (Shape: {data['early_logs'].shape})")

    print(f"Loading LATE period logs ({len(late_files)} files): {late_files[0]}...{late_files[-1]}")
    df_list_late = []
    for filename in late_files:
        try: df_list_late.append(pd.read_csv(filename))
        except Exception as e: print(f"  Error loading {filename}: {e}")
    if df_list_late:
        data['late_logs'] = pd.concat(df_list_late, ignore_index=True)
        data['late_logs']['timestamp'] = pd.to_datetime(data['late_logs']['timestamp'])
        if not data['late_logs'].empty:
            print(f"  Late logs loaded: {data['late_logs']['timestamp'].min()} to {data['late_logs']['timestamp'].max()} (Shape: {data['late_logs'].shape})")

    try:
        print(f"Loading {PARTICIPANTS_FILE}...")
        data['participants'] = pd.read_csv(PARTICIPANTS_FILE)
        print(f"Loading {TRAVEL_JOURNAL_FILE}...")
        data['travel'] = pd.read_csv(TRAVEL_JOURNAL_FILE)
        data['travel']['travelStartTime'] = pd.to_datetime(data['travel']['travelStartTime'])
        data['travel']['travelEndTime'] = pd.to_datetime(data['travel']['travelEndTime'])
        print(f"Loading {FINANCIAL_JOURNAL_FILE}...")
        data['financial'] = pd.read_csv(FINANCIAL_JOURNAL_FILE)
        data['financial']['timestamp'] = pd.to_datetime(data['financial']['timestamp'])
    except Exception as e:
        print(f"Error loading journal or participant files: {e}")
    return data

# --- Analysis Functions (Keep your existing ones) ---
def analyze_recreation_patterns(early_logs, late_logs):
    print("\n--- Hypothesis 1: Shift in 'AtRecreation' Patterns ---")
    if early_logs.empty or late_logs.empty:
        print("Insufficient log data for recreation analysis.")
        return
    results = {}
    for period_name, logs_df in [("Early", early_logs), ("Late", late_logs)]:
        recreation_logs = logs_df[logs_df['currentMode'] == 'AtRecreation'].copy()
        print(f"For {period_name} period, found {len(recreation_logs)} 'AtRecreation' log entries.")
        if recreation_logs.empty:
            results[period_name] = {'by_hour': pd.Series(dtype=int), 'by_day': pd.Series(dtype=int)}
            continue
        recreation_logs['hour_of_day'] = recreation_logs['timestamp'].dt.hour
        recreation_logs['day_of_week'] = recreation_logs['timestamp'].dt.day_name()
        hourly_counts = recreation_logs['hour_of_day'].value_counts(normalize=True).sort_index()
        daily_counts = recreation_logs['day_of_week'].value_counts(normalize=True)
        results[period_name] = {'by_hour': hourly_counts, 'by_day': daily_counts}
        print(f"\n{period_name} Period 'AtRecreation' Distribution by Hour (Top 5):\n{hourly_counts.head()}")
        print(f"\n{period_name} Period 'AtRecreation' Distribution by Day of Week (Top 5):\n{daily_counts.head()}")

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Recreation by Hour (Proportion)", "Recreation by Day of Week (Proportion)"))
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    if 'Early' in results and not results['Early']['by_hour'].empty:
        fig.add_trace(go.Bar(name='Early - Hour', x=results['Early']['by_hour'].index, y=results['Early']['by_hour'].values, marker_color='blue'), row=1, col=1)
        if not results['Early']['by_day'].empty:
             early_day_data = results['Early']['by_day'].reindex(days_order).fillna(0)
             fig.add_trace(go.Bar(name='Early - Day', x=early_day_data.index, y=early_day_data.values, marker_color='blue'), row=1, col=2)
    if 'Late' in results and not results['Late']['by_hour'].empty:
        fig.add_trace(go.Bar(name='Late - Hour', x=results['Late']['by_hour'].index, y=results['Late']['by_hour'].values, marker_color='red'), row=1, col=1)
        if not results['Late']['by_day'].empty:
            late_day_data = results['Late']['by_day'].reindex(days_order).fillna(0)
            fig.add_trace(go.Bar(name='Late - Day', x=late_day_data.index, y=late_day_data.values, marker_color='red'), row=1, col=2)
    fig.update_layout(title_text="Comparison of 'AtRecreation' Patterns (Early vs. Late Periods)", barmode='group', height=500)
    fig.update_xaxes(type='category', row=1, col=2)
    fig.show()
    print("Conclusion: Observe plots for shifts in 'AtRecreation' patterns.")


def analyze_commute_duration(travel_df, early_logs_dates, late_logs_dates):
    print("\n--- Hypothesis 3: Changes in Commuting Duration ---")
    if travel_df.empty: return print("Travel journal data empty.")
    if early_logs_dates is None or late_logs_dates is None: return print("Log date data undefined.")
    commute_travel = travel_df[travel_df['purpose'] == 'Work/Home Commute'].copy()
    if commute_travel.empty: return print("No 'Work/Home Commute' logs.")
    commute_travel['duration_minutes'] = (commute_travel['travelEndTime'] - commute_travel['travelStartTime']).dt.total_seconds() / 60
    early_min_date, early_max_date = early_logs_dates
    late_min_date, late_max_date = late_logs_dates
    print(f"  Early Period for Commute: {early_min_date} to {early_max_date}")
    print(f"  Late Period for Commute: {late_min_date} to {late_max_date}")
    early_commutes = commute_travel[(commute_travel['travelStartTime'].dt.date >= early_min_date) & (commute_travel['travelStartTime'].dt.date <= early_max_date)]
    late_commutes = commute_travel[(commute_travel['travelStartTime'].dt.date >= late_min_date) & (commute_travel['travelStartTime'].dt.date <= late_max_date)]
    avg_duration_early = early_commutes['duration_minutes'].mean() if not early_commutes.empty else np.nan
    avg_duration_late = late_commutes['duration_minutes'].mean() if not late_commutes.empty else np.nan
    print(f"Avg Commute (Early): {avg_duration_early:.2f} min ({len(early_commutes)} commutes)")
    print(f"Avg Commute (Late): {avg_duration_late:.2f} min ({len(late_commutes)} commutes)")
    if not np.isnan(avg_duration_early) and not np.isnan(avg_duration_late):
        fig = go.Figure(data=[go.Bar(name='Early', x=['Avg. Commute'], y=[avg_duration_early], marker_color='blue'), go.Bar(name='Late', x=['Avg. Commute'], y=[avg_duration_late], marker_color='red')])
        fig.update_layout(title_text="Avg Commute Duration", barmode='group', yaxis_title="Minutes")
        fig.show()
        print("Conclusion: Note difference in avg commute duration.")
    else: print("Not enough data to plot commute duration.")


def analyze_financial_spending(financial_df, early_logs_dates, late_logs_dates):
    print("\n--- Hypothesis 4: Evolution of Financial Spending ('Food', 'Recreation') ---")
    if financial_df.empty: return print("Financial journal data empty.")
    if early_logs_dates is None or late_logs_dates is None: return print("Log date data undefined.")
    categories_to_analyze = ['Food', 'Recreation']
    spending_df = financial_df[financial_df['amount'] < 0].copy()
    spending_df['amount'] = spending_df['amount'].abs()
    early_min_date, early_max_date = early_logs_dates
    late_min_date, late_max_date = late_logs_dates
    print(f"  Early Period for Spending: {early_min_date} to {early_max_date}")
    print(f"  Late Period for Spending: {late_min_date} to {late_max_date}")
    early_spending = spending_df[(spending_df['timestamp'].dt.date >= early_min_date) & (spending_df['timestamp'].dt.date <= early_max_date)]
    late_spending = spending_df[(spending_df['timestamp'].dt.date >= late_min_date) & (spending_df['timestamp'].dt.date <= late_max_date)]
    results_dict = {}
    for period_name, period_df, period_dates in [("Early", early_spending, early_logs_dates), ("Late", late_spending, late_logs_dates)]:
        if period_df.empty:
            results_dict[period_name] = pd.Series(0, index=categories_to_analyze, name='Avg Daily Spending')
            continue
        num_days_in_log_period = (period_dates[1] - period_dates[0]).days + 1
        if num_days_in_log_period <= 0: num_days_in_log_period = 1
        avg_daily_spending_series = (period_df.groupby('category')['amount'].sum() / num_days_in_log_period).reindex(categories_to_analyze).fillna(0)
        results_dict[period_name] = avg_daily_spending_series
        print(f"\nAvg Daily Spending ({period_name} Period, norm by {num_days_in_log_period} log days):\n{results_dict[period_name]}")
    if not results_dict.get("Early", pd.Series()).empty or not results_dict.get("Late", pd.Series()).empty:
        df_for_plot = pd.DataFrame(results_dict); df_for_plot.index.name = 'category'; df_plot = df_for_plot.reset_index()
        if 'category' in df_plot.columns and not df_plot.empty:
            if 'Early' not in df_plot.columns: df_plot['Early'] = 0
            if 'Late' not in df_plot.columns: df_plot['Late'] = 0
            df_plot_melted = df_plot.melt(id_vars=['category'], value_vars=['Early', 'Late'], var_name='Period', value_name='Avg Daily Spending')
            if not df_plot_melted.empty:
                fig = px.bar(df_plot_melted, x='category', y='Avg Daily Spending', color='Period', barmode='group', title="Avg Daily Spending on Food & Recreation")
                fig.show()
                print("Conclusion: Compare avg daily spending in categories.")
            else: print("Melted DataFrame for spending plot empty.")
        else: print("DataFrame for spending plot empty or missing 'category'.")
    else: print("Not enough categorized spending data to plot.")


def analyze_time_at_work(early_logs, late_logs):
    print("\n--- Hypothesis 7: Change in Time Spent 'AtWork' ---")
    if early_logs.empty or late_logs.empty: return print("Log data insufficient.")
    results = {}
    for period_name, logs_df in [("Early", early_logs), ("Late", late_logs)]:
        work_logs = logs_df[(logs_df['currentMode'] == 'AtWork') & (logs_df['timestamp'].dt.dayofweek < 5)].copy()
        if work_logs.empty: results[period_name] = np.nan; continue
        work_logs['participant_date'] = work_logs['participantId'].astype(str) + "_" + work_logs['timestamp'].dt.strftime('%Y-%m-%d')
        total_work_intervals = len(work_logs)
        num_participant_work_days = work_logs['participant_date'].nunique()
        if num_participant_work_days == 0: results[period_name] = np.nan; continue
        avg_work_minutes_per_day = (total_work_intervals * 5) / num_participant_work_days
        results[period_name] = avg_work_minutes_per_day / 60
        print(f"Avg Time 'AtWork' ({period_name}): {results[period_name]:.2f} hrs ({num_participant_work_days} p-work-days)")
    if not np.isnan(results.get("Early", np.nan)) and not np.isnan(results.get("Late", np.nan)):
        fig = go.Figure(data=[go.Bar(name='Early', x=['Avg. Time At Work'], y=[results.get("Early")], marker_color='blue'), go.Bar(name='Late', x=['Avg. Time At Work'], y=[results.get("Late")], marker_color='red')])
        fig.update_layout(title_text="Avg Weekday Time 'AtWork'", barmode='group', yaxis_title="Hours/Day")
        fig.show()
        print("Conclusion: Note difference in avg hours 'AtWork'.")
    else: print("Not enough data to plot 'AtWork' time.")


def analyze_total_travel_time(travel_df, early_logs_dates, late_logs_dates):
    print("\n--- Analysis: Overall Traveling Time ---")
    if travel_df.empty: return print("Travel journal data empty.")
    if early_logs_dates is None or late_logs_dates is None: return print("Log date data undefined.")
    early_min_date, early_max_date = early_logs_dates
    late_min_date, late_max_date = late_logs_dates
    print(f"  Early Period (Logs): {early_min_date.strftime('%Y-%m-%d')} to {early_max_date.strftime('%Y-%m-%d')}")
    print(f"  Late Period (Logs): {late_min_date.strftime('%Y-%m-%d')} to {late_max_date.strftime('%Y-%m-%d')}")
    early_travel = travel_df[(travel_df['travelStartTime'].dt.date >= early_min_date) & (travel_df['travelStartTime'].dt.date <= early_max_date)].copy()
    late_travel = travel_df[(travel_df['travelStartTime'].dt.date >= late_min_date) & (travel_df['travelStartTime'].dt.date <= late_max_date)].copy()
    total_travel_duration_early_hours = 0
    if not early_travel.empty:
        early_travel['duration'] = early_travel['travelEndTime'] - early_travel['travelStartTime']
        total_travel_duration_early_hours = early_travel['duration'].sum().total_seconds() / 3600
        print(f"Total Travel Time (Early): {total_travel_duration_early_hours:.2f} hrs ({len(early_travel)} segments)")
    else: print("No travel records for Early period.")
    total_travel_duration_late_hours = 0
    if not late_travel.empty:
        late_travel['duration'] = late_travel['travelEndTime'] - late_travel['travelStartTime']
        total_travel_duration_late_hours = late_travel['duration'].sum().total_seconds() / 3600
        print(f"Total Travel Time (Late): {total_travel_duration_late_hours:.2f} hrs ({len(late_travel)} segments)")
    else: print("No travel records for Late period.")
    if not (early_travel.empty and late_travel.empty):
        fig = go.Figure(data=[go.Bar(name='Total Travel Time', x=['Early Period', 'Late Period'], y=[total_travel_duration_early_hours, total_travel_duration_late_hours], marker_color=['blue', 'red'])])
        fig.update_layout(title_text="Total Traveling Time (Early vs. Late)", yaxis_title="Total Travel Time (Hours)")
        fig.show()
        print("Conclusion: Observe difference in total travel hours.")
    else: print("No travel data in either period to plot.")


def analyze_travel_purpose_changes(travel_df, early_logs_dates, late_logs_dates):
    print("\n--- Analysis: Changes in Travel Purpose Distribution (excluding 'Going Back to Home') ---")
    if travel_df.empty:
        print("Travel journal data is empty. Cannot analyze travel purposes.")
        return
    if early_logs_dates is None or late_logs_dates is None:
        print("Insufficient log date data (early or late period undefined) for travel purpose analysis.")
        return

    early_min_date, early_max_date = early_logs_dates
    late_min_date, late_max_date = late_logs_dates

    print(f"  Early Period for Travel Purpose: {early_min_date.strftime('%Y-%m-%d')} to {early_max_date.strftime('%Y-%m-%d')}")
    print(f"  Late Period for Travel Purpose: {late_min_date.strftime('%Y-%m-%d')} to {late_max_date.strftime('%Y-%m-%d')}")

    # Filter out the excluded purpose BEFORE further processing
    travel_df_filtered = travel_df[travel_df['purpose'] != PURPOSE_TO_EXCLUDE].copy()
    if travel_df_filtered.empty:
        print(f"No travel data left after excluding '{PURPOSE_TO_EXCLUDE}'.")
        return

    early_travel = travel_df_filtered[
        (travel_df_filtered['travelStartTime'].dt.date >= early_min_date) &
        (travel_df_filtered['travelStartTime'].dt.date <= early_max_date)
    ]
    late_travel = travel_df_filtered[
        (travel_df_filtered['travelStartTime'].dt.date >= late_min_date) &
        (travel_df_filtered['travelStartTime'].dt.date <= late_max_date)
    ]

    if early_travel.empty and late_travel.empty:
        print("No travel data found for either period to analyze purposes (after excluding).")
        return

    early_purpose_dist = early_travel['purpose'].value_counts(normalize=True) if not early_travel.empty else pd.Series(dtype=float)
    late_purpose_dist = late_travel['purpose'].value_counts(normalize=True) if not late_travel.empty else pd.Series(dtype=float)

    print("\nEarly Period Travel Purpose Distribution (Proportions, excluding 'Going Back to Home'):")
    print(early_purpose_dist)
    print("\nLate Period Travel Purpose Distribution (Proportions, excluding 'Going Back to Home'):")
    print(late_purpose_dist)

    combined_purposes = pd.concat([
        early_purpose_dist.rename("Early"),
        late_purpose_dist.rename("Late")
    ], axis=1).fillna(0)

    if not combined_purposes.empty:
        # Determine top N purposes based on their maximum proportion in either period
        combined_purposes['max_prop'] = combined_purposes.max(axis=1)
        top_purposes_to_plot = combined_purposes.sort_values(by='max_prop', ascending=False).head(TOP_N_TRAVEL_PURPOSES).index.tolist()
        
        if not top_purposes_to_plot: # Fallback if all proportions are 0
             top_purposes_to_plot = combined_purposes.head(TOP_N_TRAVEL_PURPOSES).index.tolist()


        plot_data = combined_purposes.loc[combined_purposes.index.isin(top_purposes_to_plot)].reset_index().rename(columns={'index':'purpose'})
        
        if not plot_data.empty and 'Early' in plot_data and 'Late' in plot_data : # Ensure columns exist
            plot_data_melted = plot_data.melt(id_vars='purpose', value_vars=['Early', 'Late'],
                                            var_name='Period', value_name='Proportion')

            if not plot_data_melted.empty:
                fig = px.bar(plot_data_melted, x='purpose', y='Proportion', color='Period',
                             barmode='group',
                             title=f"Distribution of Top {len(top_purposes_to_plot)} Travel Purposes (Early vs. Late, Excl. 'Going Back to Home')",
                             labels={'Proportion': 'Proportion of Trips'})
                fig.update_xaxes(categoryorder='total descending')
                fig.show()
                print(f"Conclusion: Observe changes in the proportions of the top {len(top_purposes_to_plot)} travel purposes.")
            else:
                print("Melted data for travel purpose plot is empty.")
        else:
            print("Not enough data for the selected top purposes to plot, or 'Early'/'Late' columns missing in plot_data.")
    else:
        print("No travel purpose data to plot (after excluding).")


if __name__ == "__main__":
    all_loaded_data = load_selected_logs_and_journals()

    if all_loaded_data:
        early_logs_df = all_loaded_data['early_logs']
        late_logs_df = all_loaded_data['late_logs']
        travel_df = all_loaded_data['travel']
        financial_df = all_loaded_data['financial']
        
        early_log_dates = None
        if not early_logs_df.empty:
            early_log_dates = (early_logs_df['timestamp'].dt.date.min(), early_logs_df['timestamp'].dt.date.max())
        
        late_log_dates = None
        if not late_logs_df.empty:
            late_log_dates = (late_logs_df['timestamp'].dt.date.min(), late_logs_df['timestamp'].dt.date.max())

        # --- Run Analyses ---
        analyze_recreation_patterns(early_logs_df, late_logs_df)
        analyze_commute_duration(travel_df, early_log_dates, late_log_dates)
        analyze_financial_spending(financial_df, early_log_dates, late_log_dates)
        analyze_time_at_work(early_logs_df, late_logs_df)
        analyze_total_travel_time(travel_df, early_log_dates, late_log_dates)
        analyze_travel_purpose_changes(travel_df, early_log_dates, late_log_dates)

        print("\n--- Analysis Complete ---")
        print(f"This analysis used {NUM_FILES_PER_PERIOD} log files for 'early' and 'late' periods respectively.")
    else:
        print("Failed to load sufficient initial data. Exiting.")
