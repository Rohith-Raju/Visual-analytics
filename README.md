# Visual Project

This project contains Python scripts and a subdirectory primarily focused on data analysis, visualization, and potentially image processing, with a significant portion dedicated to analyzing datasets from the VAST Challenge 2022.

## Project Directory (`visual/Project/`)

The `visual/Project/` directory contains scripts that perform more complex analyses, largely centered around the VAST Challenge 2022 datasets.

### `visual/Project/Question1.py`

- **Description:** This script visualizes geographical data from the VAST Challenge 2022. It reads CSV files for buildings, apartments, pubs, restaurants, and employers, parses their location data (polygons and points), and creates an interactive map using Plotly.
- **Functionality:**
  - Displays building footprints (Commercial, Residential, School) with distinct colors.
  - Plots locations of apartments, restaurants, and pubs as markers.
  - Includes hover-over information for points of interest (e.g., rental cost, food cost, pub ID).
  - Features toggle buttons to switch between different views: an initial view with basic markers, and characteristic views where markers are colored/styled based on attributes (e.g., apartment rental cost, restaurant food cost, pub hourly cost).
  - Adds a heatmap layer for employer density.
  - Generates and saves a base map image (`BaseMap.png`).
- **Usage:** The script reads data from specific CSV file paths (e.g., `VAST-Challenge-2022/Datasets/Attributes/Buildings.csv`). Ensure these files are present in the expected locations. It then generates and displays an interactive Plotly figure.

### `visual/Project/Question2.1.py`

- **Description:** This script processes and visualizes travel trajectories of participants from the VAST Challenge 2022 dataset. It focuses on specific days of interest and the top travel purposes.
- **Functionality:**
  - Loads travel journal data and participant activity logs.
  - Filters data for "Transport" mode and specific days (e.g., Tuesday, Saturday).
  - Focuses on the top 3 most frequent travel purposes.
  - Aggregates trajectory points for each trip, associating them with purpose, day, and time of day (Day/Night).
  - Uses Plotly to create an interactive scatter plot of trajectories, color-coded by purpose.
  - Provides dropdown menus and buttons to filter the displayed trajectories by day, travel purpose, and time of day.
- **Usage:** Requires `TravelJournal.csv` and `ParticipantStatusLogs1.csv` from the VAST Challenge 2022 dataset. Displays an interactive Plotly graph.

### `visual/Project/Question2.2.py`

- **Description:** This script generates traffic density heatmaps based on participant activity logs from the VAST Challenge 2022. It creates a base map from attribute data and overlays heatmaps showing traffic density for different days and 3-hour intervals.
- **Functionality:**
  - Loads attribute data (buildings, apartments, etc.) to create a static base map of city locations.
  - Processes multiple activity log files to extract "Transport" mode locations and timestamps.
  - Aggregates traffic data into 3-hour intervals for each day of the week.
  - Generates 2D histogram heatmaps using Plotly for each day and time interval combination.
  - Provides a dropdown menu to select the day and a slider to select the 3-hour time interval, updating the heatmap dynamically.
- **Usage:** Expects VAST Challenge 2022 datasets in a `VAST-Challenge-2022/Datasets/` subdirectory. Displays an interactive Plotly figure with heatmaps.

### `visual/Project/Question3.py`

- **Description:** This script visualizes an individual participant's daily routine by combining data from activity logs, travel journals, and financial journals, likely from the VAST Challenge 2022.
- **Functionality:**
  - Loads data from multiple activity log files, participant attributes, travel journal, financial journal, and check-in journal.
  - Allows selection of specific participant IDs and a target date.
  - Processes activity logs for the selected participant and date to create segments representing different modes (e.g., AtHome, AtWork, Transport).
  - Integrates travel segments from the travel journal, overlaying them with purpose.
  - Adds financial transactions (expenses/income) from the financial journal as markers on the timeline.
  - Generates a Plotly timeline (Gantt-like chart) showing the participant's activities throughout the day, with color-coding for different modes/travel.
- **Usage:** Requires various CSV files from the VAST Challenge 2022 dataset. The script has `SELECTED_PARTICIPANT_IDS` and `TARGET_DATE_STR` variables that can be modified to analyze different participants and dates.

### `visual/Project/Question4.py`

- **Description:** This script performs a comparative analysis of participant behavior between an "early" and "late" period, using data from the VAST Challenge 2022. It investigates several hypotheses related to changes in daily patterns.
- **Functionality:**
  - Loads a defined number of log files from the beginning and end of the available dataset to represent "early" and "late" periods.
  - Analyzes and compares:
    - 'AtRecreation' patterns (distribution by hour and day of the week).
    - Commuting duration for 'Work/Home Commute' purpose.
    - Financial spending patterns, particularly for 'Food' and 'Recreation'.
    - Time spent 'AtWork' on weekdays.
    - Total travel time.
    - Changes in the distribution of travel purposes (excluding 'Going Back to Home').
  - Generates various Plotly bar charts and subplots to visualize these comparisons.
- **Usage:** Expects VAST Challenge 2022 datasets. The `NUM_FILES_PER_PERIOD` variable controls how many log files define the early and late periods.

## Python Scripts (`visual/`)

These are standalone Python scripts found in the root `visual` directory.

### `visual/lab2.py`

- **Description:** This script analyzes gaze data from a TSV file. It performs K-Means clustering on gaze points and visualizes the clusters, their usage over time, and transitions between them.
- **Functionality:**
  - Loads gaze data from a TSV file (`data.tsv`).
  - Performs K-Means clustering on X and Y gaze coordinates to identify regions of interest.
  - Visualizes the raw gaze data and the clustered data with centroids using Plotly scatter plots.
  - Analyzes cluster usage over time by segmenting data into time intervals and plotting fixation counts per cluster.
  - Identifies heavily used regions (clusters) overall.
  - Calculates and visualizes transition patterns between clusters using a Sankey diagram.
  - Creates an animated 3D scatter plot (space-time cube) to show cluster transitions over time windows.
- **Usage:** Requires a `data.tsv` file in the same directory. Ensure Python libraries like `pandas`, `scikit-learn`, and `plotly` are installed. Run the script, and it will display several interactive plots.

### `visual/lab3.py`

- **Description:** This script extracts various visual features from a collection of images and then ranks these images based on their similarity to a chosen image for each feature.
- **Functionality:**
  - Loads images from a specified folder (`./Lab3.1`).
  - Extracts features for each image:
    - Average color of the entire image.
    - Average color of a central patch.
    - Overall luminance.
    - Edge density (using Canny edge detection).
    - Concatenated color histograms (for B, G, R channels).
  - Computes a distance matrix (Euclidean distance) between images for each feature type.
  - Ranks images based on their distance to a pre-selected image for each feature.
  - Prints the ranked list of images and their distances for each feature.
- **Usage:** Requires a folder named `Lab3.1` containing `.jpg` images in the same directory as the script. Ensure `opencv-python` (`cv2`), `numpy`, and `scipy` are installed. Run the script to see the ranked output in the console.

## Getting Started

[Provide instructions on how to set up and run the project, if applicable. Include any dependencies or environment setup.]

**General Dependencies:**

- Python 3.x
- pandas
- scikit-learn
- plotly
- opencv-python
- numpy
- scipy
- tqdm (used in some Project scripts)
- shapely (used in `Project/Question1.py`)

You can typically install these using pip:
`pip install pandas scikit-learn plotly opencv-python numpy scipy tqdm shapely`
