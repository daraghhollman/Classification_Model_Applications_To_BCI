"""
A test file, which loads the random forest classifer for the bow shock, and uses an overlapping sliding window to make predictions of a time series
"""

import datetime as dt
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.ensemble
import scipy.stats
from tqdm import tqdm
from hermpy import mag, utils, trajectory, boundaries

# Import time series
start = dt.datetime(2011, 8, 31, 16, 0)
end = dt.datetime(2011, 8, 31, 18, 0)
data = mag.Load_Between_Dates(
    utils.User.DATA_DIRECTORIES["MAG_FULL"], start, end, average=None
)

# Load Model
with open(
    "/home/daraghhollman/Main/Work/mercury/DataSets/bow_shock_random_forest", "rb"
) as file:
    random_forest: sklearn.ensemble.RandomForestClassifier = pickle.load(file)

# Grazing angle calculation requires a reference boundary. This might be an issue
boundary = "bow shock"

window_size = 30  # seconds. How large of a window to feed to the random forest
step_size = 5  # seconds. How far should the window jump each time

window_start = data["date"].iloc[0]
window_end = data["date"].iloc[window_size]

# While the end of the moving window is within the data range
# Collect samples
region_predictions = []
region_probabilities = []
window_dates = []
with tqdm(total=((data["date"].iloc[-1] - data["date"].iloc[0]).total_seconds() / step_size)) as progress_bar:
    while window_end < data["date"].iloc[-1]:

        window_dates.append(window_start + (window_end - window_start))

        # Get features of the distribution
        data_section = data.loc[data["date"].between(window_start, window_end)]
        features = dict()

        for component in ["|B|", "Bx", "By", "Bz"]:
            features[f"Mean {component}"] = (np.mean(data_section[component]),)
            features[f"Median {component}"] = (np.median(data_section[component]),)
            features[f"Standard Deviation {component}"] = (np.std(data_section[component]),)
            features[f"Skew {component}"] = (scipy.stats.skew(data_section[component]),)
            features[f"Kurtosis {component}"] = scipy.stats.kurtosis(
                data_section[component]
            )

        middle_data_point = data_section.iloc[len(data_section) // 2]
        middle_position = [
            middle_data_point["X MSM' (radii)"],
            middle_data_point["Y MSM' (radii)"],
            middle_data_point["Z MSM' (radii)"],
        ]
        middle_features = [
            "X MSM' (radii)",
            "Y MSM' (radii)",
            "Z MSM' (radii)",
        ]
        for feature in middle_features:
            features[feature] = middle_data_point[feature]

        features["Latitude (deg.)"] = trajectory.Latitude(middle_position)
        features["Magnetic Latitude (deg.)"] = trajectory.Magnetic_Latitude(middle_position)
        features["Local Time (hrs)"] = trajectory.Local_Time(middle_position)
        features["Heliocentric Distance (AU)"] = trajectory.Get_Heliocentric_Distance(
            middle_data_point["date"]
        )

        # Apply random forest and get prediction
        X = pd.DataFrame(features)
        column_names = list(X.columns.values)
        column_names.sort()
        X = X[column_names]

        prediction = random_forest.predict(X)[0]
        probabilities = random_forest.predict_proba(X)[0]

        region_predictions.append(prediction)
        region_probabilities.append(probabilities)

        # Move the window
        window_start = window_end
        window_end += dt.timedelta(seconds=step_size)

        progress_bar.update(1)


fig, ax = plt.subplots()

ax.plot(data["date"], data["|B|"], color="black")

colours = {"Solar Wind": "cornflowerblue", "Magnetosheath": "indianred"}

solar_wind_probability = np.array(region_probabilities)[:, 1]

uncertainty = False
uncertainty_size = 0
if uncertainty:

    solar_wind = solar_wind_probability > 0.5 + uncertainty_size
    magnetosheath = solar_wind_probability < 0.5 - uncertainty_size
    uncertain_region = (solar_wind_probability > 0.5 - uncertainty_size) & (
        solar_wind_probability < 0.5 + uncertainty_size
    )
    sw_label = f"P(SW) > {0.5 + uncertainty_size}"
    ms_label = f"P(SW) < {0.5 - uncertainty_size}"
    uncertain_label = f"{0.5 - uncertainty_size} < P(SW) < {0.5 + uncertainty_size}"

else:
    solar_wind = solar_wind_probability > 0.5
    magnetosheath = solar_wind_probability < 0.5

    sw_label = f"P(SW) > {0.5}"
    ms_label = f"P(SW) < {0.5}"

# Add region shading
# Iterate through each window
for i in range(len(window_dates) - 1):

    if i == 0:
        continue

    if uncertainty:
        if solar_wind[i]:
            ax.axvspan(
                window_dates[i] - (window_dates[i] - window_dates[i-1]) / 2,
                window_dates[i] + (window_dates[i + 1] - window_dates[i]) / 2,
                color="cornflowerblue",
                alpha=0.3,
                label=sw_label,
            )
            sw_label = ""

        elif magnetosheath[i]:
            ax.axvspan(
                window_dates[i] - (window_dates[i] - window_dates[i-1]) / 2,
                window_dates[i] + (window_dates[i + 1] - window_dates[i]) / 2,
                color="indianred",
                alpha=0.3,
                label=ms_label,
            )
            ms_label = ""

        else:
            ax.axvspan(
                window_dates[i] - (window_dates[i] - window_dates[i-1]) / 2,
                window_dates[i] + (window_dates[i + 1] - window_dates[i]) / 2,
                color="lightgrey",
                alpha=0.3,
                label=uncertain_label,
            )
            uncertain_label = ""

    else:
        if solar_wind[i]:
            ax.axvspan(
                window_dates[i] - (window_dates[i] - window_dates[i-1]) / 2,
                window_dates[i] + (window_dates[i + 1] - window_dates[i]) / 2,
                color="cornflowerblue",
                alpha=0.3,
                label=sw_label,
            )
            sw_label = ""

        else:
            ax.axvspan(
                window_dates[i] - (window_dates[i] - window_dates[i-1]) / 2,
                window_dates[i] + (window_dates[i + 1] - window_dates[i]) / 2,
                color="indianred",
                alpha=0.3,
                label=ms_label,
            )
            ms_label = ""


probability_ax = ax.twinx()
probability_ax.plot(
    window_dates, solar_wind_probability, color="blue", zorder=10, alpha=0.5, lw=2
)

ax.legend()

ax.set_ylabel("|B| [nT]")
probability_ax.set_ylabel("Solar Wind Probability")

ax.set_title(f"Random Forest Application (Overlapping Sliding Window)\nWindow Size: {window_size} s    Step Size: {step_size} s")


# Add boundary crossing intervals
crossings = boundaries.Load_Crossings(utils.User.CROSSING_LISTS["Philpott"])
boundaries.Plot_Crossing_Intervals(ax, start, end, crossings, color="black")

plt.show()
