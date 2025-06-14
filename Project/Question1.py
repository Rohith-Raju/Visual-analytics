import pandas as pd
import plotly.graph_objects as go
from shapely.geometry import Point
import numpy as np

building_csv_path = "VAST-Challenge-2022/Datasets/Attributes/Buildings.csv"
apartment_csv_path = "VAST-Challenge-2022/Datasets/Attributes/Apartments.csv"
pub_csv_path = "VAST-Challenge-2022/Datasets/Attributes/Pubs.csv"
restaurant_csv_path = "VAST-Challenge-2022/Datasets/Attributes/Restaurants.csv"
employers_csv_path = "VAST-Challenge-2022/Datasets/Attributes/Employers.csv"  # Assuming the file is named "Employers.csv"

buildings_df = pd.read_csv(building_csv_path)
apartments_df = pd.read_csv(apartment_csv_path)
pubs_df = pd.read_csv(pub_csv_path)
restaurants_df = pd.read_csv(restaurant_csv_path)
employers_df = pd.read_csv(employers_csv_path)

building_polygons_coords = []
building_types = []
for poly_str, building_type in zip(
    buildings_df["location"], buildings_df["buildingType"]
):
    coords = poly_str.replace("POLYGON ((", "").replace("))", "").split(", ")
    points = []
    for coord in coords:
        try:
            x, y = map(float, coord.strip().split())
            points.append((x, y))
        except ValueError:
            continue
    if len(points) >= 3:
        # Normalize building type
        bt = building_type.strip().lower()
        if bt.startswith("resid"):
            bt = "Residental"
        elif bt.startswith("comm"):
            bt = "Commercial"
        elif bt.startswith("school"):
            bt = "School"
        building_polygons_coords.append(points)
        building_types.append(bt)

apartment_points_coords = []
apartment_ids = []
rental_costs = []
for point_str, apartment_id, rental_cost in zip(
    apartments_df["location"], apartments_df["apartmentId"], apartments_df["rentalCost"]
):
    coords = point_str.replace("POINT (", "").replace(")", "").split()
    try:
        x, y = map(float, coords)
        apartment_points_coords.append((x, y))
        apartment_ids.append(apartment_id)
        rental_costs.append(rental_cost)
    except ValueError:
        continue

restaurant_points = []
restaurant_ids = []
restaurant_hourly = []
for point_str, rest_id, hourly in zip(
    restaurants_df["location"], restaurants_df["restaurantId"], restaurants_df["foodCost"]
):
    coords = point_str.replace("POINT (", "").replace(")", "").split()
    try:
        x, y = map(float, coords)
        restaurant_points.append((x, y))
        restaurant_ids.append(rest_id)
        restaurant_hourly.append(hourly)
    except ValueError:
        continue

pub_points_coords = []
pub_ids = []
pub_hourly_costs = []
for point_str, pub_id, hourly_cost in zip(
    pubs_df["location"], pubs_df["pubId"], pubs_df["hourlyCost"]
):
    coords = point_str.replace("POINT (", "").replace(")", "").split()
    try:
        x, y = map(float, coords)
        pub_points_coords.append((x, y))
        pub_ids.append(pub_id)
        pub_hourly_costs.append(hourly_cost)
    except ValueError:
        continue

employer_points_coords = []
building_ids = []
for point_str, building_id in zip(
    employers_df["location"], employers_df["buildingId"]
):
    coords = point_str.replace("POINT (", "").replace(")", "").split()
    try:
        x, y = map(float, coords)
        employer_points_coords.append((x, y))
        building_ids.append(building_id)
    except ValueError:
        continue

apartment_nearest_pub_distance = []
pub_points_shapely = [Point(x, y) for x, y in pub_points_coords]
for x, y in apartment_points_coords:
    apt_point = Point(x, y)
    if pub_points_shapely:
        min_dist = min(
            apt_point.distance(pub_point) for pub_point in pub_points_shapely
        )
    else:
        min_dist = np.inf
    apartment_nearest_pub_distance.append(min_dist)

fig = go.Figure()

building_type_colors = {
    "Commercial": "blue",
    "Residental": "green",
    "School": "orange",
}

for polygon_coords, building_type in zip(building_polygons_coords, building_types):
    x_coords, y_coords = zip(*polygon_coords)
    color = building_type_colors.get(building_type, "gray")
    fig.add_trace(
        go.Scatter(
            x=x_coords,
            y=y_coords,
            fill="toself",
            mode="lines",
            line=dict(color=color, width=1),
            opacity=0.5,
            name=building_type,
            hoverinfo="skip",
            showlegend=False,
            visible=True,
        )
    )

fig.write_image("./BaseMap.png")

x_apartments, y_apartments = zip(*apartment_points_coords)
initial_apartment_hover_text = [
    f"Apartment ID: {apartment_id}<br>Rental Cost: ${rental_cost}"
    for apartment_id, rental_cost in zip(apartment_ids, rental_costs)
]
fig.add_trace(
    go.Scattergl(
        x=x_apartments,
        y=y_apartments,
        mode="markers",
        name="Apartments (Initial)",
        marker=dict(color="red", size=6),
        text=initial_apartment_hover_text,
        hoverinfo="text",
        showlegend=True,
        visible=True,
        legendgroup="apartments",
    )
)
initial_apartment_trace_index = len(fig.data) - 1

characteristic_apartment_hover_text = [
    f"Apartment ID: {apartment_ids[i]}<br>Rental Cost: ${rental_costs[i]}<br>"
    f"Dist to Nearest Pub: {apartment_nearest_pub_distance[i]:.2f} units"
    if apartment_nearest_pub_distance[i] != np.inf
    else f"Apartment ID: {apartment_ids[i]}<br>Rental Cost: ${rental_costs[i]}<br>No pubs found nearby"
    for i in range(len(apartment_ids))
]
fig.add_trace(
    go.Scattergl(
        x=x_apartments,
        y=y_apartments,
        mode="markers",
        name="Apartments (by Rent)",
        marker=dict(
            color=rental_costs,
            colorscale="Jet",
            size=6,
            colorbar=dict(
                title="Rental Cost",
                thickness=15,
                x=1.02,
                y=0.5,
                xanchor="left",
                yanchor="middle",
            ),
            showscale=True,
        ),
        text=characteristic_apartment_hover_text,
        hoverinfo="text",
        showlegend=True,
        visible=False,
        legendgroup="apartments",
    )
)
characteristic_apartment_trace_index = len(fig.data) - 1

# --- Restaurants: Initial (purple dots, no colorbar) ---
x_rest, y_rest = zip(*restaurant_points)
initial_restaurant_hover_text = [f"Restaurant ID: {restid}" for restid in restaurant_ids]
fig.add_trace(
    go.Scattergl(
        x=x_rest,
        y=y_rest,
        mode="markers",
        name="Restaurants (Initial)",
        marker=dict(color="purple", size=7),
        text=initial_restaurant_hover_text,
        hoverinfo="text",
        showlegend=True,
        visible=True,
        legendgroup="restaurants",
    )
)
initial_restaurant_trace_index = len(fig.data) - 1

characteristic_restaurant_hover_text = [
    f"Restaurant ID: {restid}<br>Food Cost: ${hourly}"
    for restid, hourly in zip(restaurant_ids, restaurant_hourly)
]
fig.add_trace(
    go.Scattergl(
        x=x_rest,
        y=y_rest,
        mode="markers",
        name="Restaurants (by Food Cost)",
        marker=dict(
            color=restaurant_hourly,
            colorscale="Plasma",
            size=7,
            colorbar=dict(
                title="Food Cost",
                thickness=15,
                x=1.02,
                y=0.5,
                xanchor="left",
                yanchor="middle",
            ),
            showscale=True,
        ),
        text=characteristic_restaurant_hover_text,
        hoverinfo="text",
        showlegend=True,
        visible=False,
        legendgroup="restaurants",
    )
)
characteristic_restaurant_trace_index = len(fig.data) - 1

# --- Pubs: Initial (black dots) ---
x_pubs, y_pubs = zip(*pub_points_coords)
initial_pub_hover_text = [
    f"Pub ID: {pubid}<br>Hourly Cost: ${hourlyRate}"
    for pubid, hourlyRate in zip(pub_ids, pub_hourly_costs)
]
fig.add_trace(
    go.Scattergl(
        x=x_pubs,
        y=y_pubs,
        mode="markers",
        name="Pubs (Initial)",
        marker=dict(color="black", size=6),
        text=initial_pub_hover_text,
        hoverinfo="text",
        showlegend=True,
        visible=True,
        legendgroup="pubs",
    )
)
initial_pub_trace_index = len(fig.data) - 1

# --- Pubs: Characteristic (diamond, color by cost) ---
characteristic_pub_hover_text = [
    f"Pub ID: {pubid}<br>Hourly Cost: ${hourlyRate}"
    for pubid, hourlyRate in zip(pub_ids, pub_hourly_costs)
]
fig.add_trace(
    go.Scattergl(
        x=x_pubs,
        y=y_pubs,
        mode="markers",
        name="Pubs (by Cost)",
        marker=dict(
            color=pub_hourly_costs
            if pub_hourly_costs and any(pub_hourly_costs)
            else "black",
            colorscale="Reds",
            size=8,
            symbol="diamond",
            colorbar=dict(
                title="Pub Hourly Cost",
                thickness=15,
                x=1.08,
                y=0.5,
                xanchor="left",
                yanchor="middle",
            ),
            showscale=True,
        ),
        text=characteristic_pub_hover_text,
        hoverinfo="text",
        showlegend=True,
        visible=False,
        legendgroup="pubs",
    )
)
characteristic_pub_trace_index = len(fig.data) - 1

# --- Employers: Heatmap (NEW) ---
x_employers, y_employers = zip(*employer_points_coords)
# Add a density heatmap for employer locations
employer_heatmap = go.Histogram2d(
    x=x_employers,
    y=y_employers,
    colorscale="YlOrRd",
    colorbar=dict(
        title="Employer Density",
        thickness=15,
        x=1.08,
        y=0.5,
        xanchor="left",
        yanchor="middle",
    ),
    showscale=True,
    name="Employers (Heatmap)",
    visible=False,  # Initially hidden
    legendgroup="employers",
    hovertemplate="Employer Density: %{z}<extra></extra>",
)

fig.add_trace(employer_heatmap)
employer_heatmap_trace_index = len(fig.data) - 1

# --- Building Type Legend (as lines, always visible) ---
for building_type, color in building_type_colors.items():
    if (
        building_type in building_types
        or building_type in ["Commercial", "Residental", "School"]
    ):
        fig.add_trace(
            go.Scatter(
                x=[None, None],
                y=[None, None],
                mode="lines",
                line=dict(color=color, width=4),
                name=f"{building_type} Buildings",
                showlegend=True,
                visible=True,
                legendgroup="buildings",
            )
        )

# --- Toggle Buttons ---
toggle_trace_indices = [
    initial_apartment_trace_index,  # 0
    characteristic_apartment_trace_index,  # 1
    initial_restaurant_trace_index,  # 2
    characteristic_restaurant_trace_index,  # 3
    initial_pub_trace_index,  # 4
    characteristic_pub_trace_index,  # 5
    employer_heatmap_trace_index,  # 6 (now the heatmap)
]
updatemenu = go.layout.Updatemenu(
    type="buttons",
    direction="left",
    x=1.0,
    y=1.15,
    xanchor="right",
    yanchor="top",
    pad={"r": 10, "t": 10},
    buttons=[
        dict(
            label="Initial View",
            method="update",
            args=[
                {
                    "visible": [
                        True,
                        False,
                        True,
                        False,
                        True,
                        False,
                        False,  # Heatmap hidden
                    ],
                    "marker.showscale": [
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                    ],
                },
                {"legend": {"x": 1.0, "y": 1.0, "xanchor": "left", "yanchor": "top"}},
                toggle_trace_indices,
            ],
        ),
        dict(
            label="Apartments",
            method="update",
            args=[
                {
                    "visible": [
                        False,
                        True,
                        False,
                        False,
                        False,
                        False,
                        False,
                    ],
                    "marker.showscale": [
                        False,
                        True,
                        False,
                        False,
                        False,
                        False,
                        False,
                    ],
                },
                {"legend": {"x": 0.0, "y": 1.0, "xanchor": "left", "yanchor": "top"}},
                toggle_trace_indices,
            ],
        ),
        dict(
            label="Restaurants",
            method="update",
            args=[
                {
                    "visible": [
                        False,
                        False,
                        False,
                        True,
                        False,
                        False,
                        False,
                    ],
                    "marker.showscale": [
                        False,
                        False,
                        False,
                        True,
                        False,
                        False,
                        False,
                    ],
                },
                {"legend": {"x": 0.0, "y": 1.0, "xanchor": "left", "yanchor": "top"}},
                toggle_trace_indices,
            ],
        ),
        dict(
            label="Pubs",
            method="update",
            args=[
                {
                    "visible": [
                        False,
                        False,
                        False,
                        False,
                        False,
                        True,
                        False,
                    ],
                    "marker.showscale": [
                        False,
                        False,
                        False,
                        False,
                        False,
                        True,
                        False,
                    ],
                },
                {"legend": {"x": 0.0, "y": 1.0, "xanchor": "left", "yanchor": "top"}},
                toggle_trace_indices,
            ],
        ),
        dict(
            label="Employers",
            method="update",
            args=[
                {
                    "visible": [
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        True,  # Only heatmap visible
                    ],
                    "marker.showscale": [
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        True,  # Show colorbar for heatmap
                    ],
                },
                {"legend": {"x": 0.0, "y": 1.0, "xanchor": "left", "yanchor": "top"}},
                toggle_trace_indices,
            ],
        ),
    ],
)

# --- Layout ---
fig.update_layout(
    title="City Map: Buildings, Apartments, Pubs, and Restaurants",
    xaxis_title="X Coordinate",
    yaxis_title="Y Coordinate",
    legend_title="Legend",
    template="plotly_white",
    hovermode="closest",
    yaxis=dict(scaleanchor="x", scaleratio=1),
    updatemenus=[updatemenu],
    margin=dict(l=20, r=20, t=100, b=20),
    legend=dict(
        x=1.0,
        y=1.0,
        xanchor="left",
        yanchor="top",
        bgcolor="rgba(255, 255, 255, 0.6)",
        bordercolor="rgba(0, 0, 0, 0.2)",
        borderwidth=1,
    ),
)

fig.show()