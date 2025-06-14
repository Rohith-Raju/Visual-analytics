import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import sys  


FILE_PATH = 'data.tsv'


COLUMN_NAMES = [
    'RecordingTimestamp',
    'FixationIndex',
    'GazeEventDuration(mS)',
    'GazePointIndex',
    'GazePointX(px)',
    'GazePointY(px)',
]
X_COL = 'GazePointX(px)'
Y_COL = 'GazePointY(px)'

data = pd.read_csv(FILE_PATH, sep='\s+', header=0, names=COLUMN_NAMES)
print(f"Data loaded successfully from '{FILE_PATH}'.")
print(f"Shape: {data.shape}")
print("First 5 rows:")
print(data.head())
print("-" * 30)

if X_COL not in data.columns or Y_COL not in data.columns:
    print(
        f"Error: Required columns '{X_COL}' or '{Y_COL}' not found in the data."
    )
    print(f"Available columns: {list(data.columns)}")
    sys.exit(1)

print(X_COL)
print(Y_COL)

coordinates = data[[X_COL, Y_COL]].values

N_CLUSTERS = 3
RANDOM_STATE = 10


fig = px.scatter(
    data,
    x=X_COL,
    y=Y_COL,
    title='Gaze Data',
    labels={X_COL: 'X Coordinate', Y_COL: 'Y Coordinate'},
)
fig.show()


kmeans = KMeans(
    n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10
)  

data['Cluster'] = kmeans.fit_predict(coordinates)


fig = px.scatter(
    data,
    x=X_COL,
    y=Y_COL,
    color='Cluster',
    title='K-Means ',
    labels={X_COL: 'X Coordinate', Y_COL: 'Y Coordinate', 'Cluster': 'Cluster'},
)


fig.add_trace(
    go.Scatter(
        x=kmeans.cluster_centers_[:, 0],
        y=kmeans.cluster_centers_[:, 1],
        mode='markers',
        marker=dict(size=12, color='black', symbol='x'),
        name='Centroids',
    )
)

fig.show()



data['Time'] = pd.to_datetime(
    data['RecordingTimestamp'], unit='ms'
) 


time_interval = 'x  '  
data['TimeInterval'] = data['Time'].dt.floor(time_interval)


cluster_usage = (
    data.groupby(['TimeInterval', 'Cluster'])
    .size()
    .unstack(fill_value=0)
)  

print( data.groupby(['TimeInterval', 'Cluster'])
    .size()
    .unstack(fill_value=0))


cluster_totals = data['Cluster'].value_counts().sort_values(ascending=False)
print("\nHeavily Used Regions (Overall):")
print(cluster_totals)


cluster_usage_df = cluster_usage.reset_index()  
cluster_usage_melted = cluster_usage_df.melt(
    id_vars='TimeInterval',
    var_name='Cluster',
    value_name='FixationCount',
)  

fig = px.line(
    cluster_usage_melted,
    x='TimeInterval',
    y='FixationCount',
    color='Cluster',
    title='Cluster Usage',
    labels={'TimeInterval': 'Time', 'FixationCount': 'Number of points'},
)
fig.show()



cluster_start_end = data.groupby('Cluster')['Time'].agg(['min', 'max'])




data['PreviousCluster'] = data['Cluster'].shift(1)


data = data.dropna()


transition_counts = (
    data.groupby(['PreviousCluster', 'Cluster'])
    .size()
)


print("\nFrequent Transitions Between Clusters:")
print(transition_counts)



transition_matrix = (
    data.groupby(['PreviousCluster', 'Cluster']).size().unstack(fill_value=0)
)
print("\nTransition Matrix:")
print(transition_matrix)


labels = [f'Cluster {i}' for i in range(N_CLUSTERS)]

source = []
target = []
value = []
for i in range(N_CLUSTERS):
    for j in range(N_CLUSTERS):
        count = transition_matrix.iloc[i, j]
        if count > 0:
            source.append(i)
            target.append(j)
            value.append(count)


link = {'source': source, 'target': target, 'value': value}
node = {'label': labels}
sankey = go.Sankey(link=link, node=node)
fig = go.Figure(sankey)
fig.update_layout(title_text="Cluster Transition", font_size=10)
fig.show()





window = '10S'
data['TransitionWindow'] = data['Time'].dt.floor(window)


cluster_ids = list(range(N_CLUSTERS))
all_pairs = [(i, j) for i in cluster_ids for j in cluster_ids]


cube_records = []
for win in sorted(data['TransitionWindow'].unique()):
    window_data = data[data['TransitionWindow'] == win]
    if len(window_data) < 2:
        continue
    tm = (
        window_data.groupby(['PreviousCluster', 'Cluster'])
        .size()
        .reindex(all_pairs, fill_value=0)
        .reset_index()
    )
    tm.columns = ['From', 'To', 'Count']
    tm['TimeWindow'] = win
    cube_records.append(tm)

cube_df = pd.concat(cube_records, ignore_index=True)

fig = px.scatter_3d(
    cube_df,
    x="From",
    y="To",
    z="TimeWindow",
    color="Count",
    size="Count",
    animation_frame="TimeWindow",
    range_color=[0, cube_df['Count'].max()],
    title="Space-Time Cube of Cluster Transitions (Animated)",
    labels={"From": "From Cluster", "To": "To Cluster", "TimeWindow": "Time Window"},
    color_continuous_scale="Viridis",
)

fig.update_traces(marker=dict(symbol='circle', opacity=0.8))
fig.update_layout(scene=dict(
    xaxis=dict(dtick=1, title="From Cluster"),
    yaxis=dict(dtick=1, title="To Cluster"),
    zaxis=dict(title="Time Window"),
))
fig.show()
