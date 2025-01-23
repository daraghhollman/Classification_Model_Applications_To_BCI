"""
A test file, which loads the random forest classifer for the bow shock, and uses an overlapping sliding window to make predictions of a time series
"""

import multiprocessing
import datetime as dt
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.ensemble
import scipy.stats
from tqdm import tqdm
from hermpy import mag, utils, trajectory, boundaries, plotting


crossings = boundaries.Load_Crossings(utils.User.CROSSING_LISTS["Philpott"])[96:]
crossings = crossings.loc[crossings["Type"].str.contains("BS")]

time_buffer = dt.timedelta(minutes=10)

# Load Model
print("Loading model")
with open(
    "/home/daraghhollman/Main/Work/mercury/DataSets/bow_shock_random_forest", "rb"
) as file:
    random_forest: sklearn.ensemble.RandomForestClassifier = pickle.load(file)

for i, crossing in crossings.iterrows():
    print(f"Processing crossing {i}")
    # Import time series
    start = crossing["Start Time"] - time_buffer
    end = crossing["End Time"] + time_buffer

    print(f"Loading data between {start} and {end}")
    data = mag.Load_Between_Dates(
        utils.User.DATA_DIRECTORIES["MAG_FULL"], start, end, average=None
    )

    # Grazing angle calculation requires a reference boundary. This might be an issue
    boundary = "bow shock"

    window_size = 10  # seconds. How large of a window to feed to the random forest
    step_size = 5  # seconds. How far should the window jump each time

    windows = [
        (window_start, window_start + dt.timedelta(seconds=window_size))
        for window_start in pd.date_range(start=start, end=end, freq=f"{step_size}s")
    ]
    window_centres = [window_start + (window_end - window_start) / 2 for window_start, window_end in windows]

    def Get_Window_Features(input):
        window_start, window_end = input

        data_section = data.loc[data["date"].between(window_start, window_end)]

        if len(data_section) == 0:
            return 

        # Find features
        features = dict()
        for component in ["|B|", "Bx", "By", "Bz"]:
            component_data = data_section[component]
            features.update(
                {
                    f"Mean {component}": np.mean(component_data),
                    f"Median {component}": np.median(component_data),
                    f"Standard Deviation {component}": np.std(component_data),
                    f"Skew {component}": scipy.stats.skew(component_data),
                    f"Kurtosis {component}": scipy.stats.kurtosis(component_data),
                }
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

        features.update(
            {
                "Latitude (deg.)": trajectory.Latitude(middle_position),
                "Magnetic Latitude (deg.)": trajectory.Magnetic_Latitude(
                    middle_position
                ),
                "Local Time (hrs)": trajectory.Local_Time(middle_position),
                "Heliocentric Distance (AU)": trajectory.Get_Heliocentric_Distance(
                    middle_data_point["date"]
                ),
            }
        )

        # Prediction
        X = pd.DataFrame([features])
        column_names = list(X.columns.values)
        column_names.sort()
        X = X[column_names]

        return X

    samples = []
    missing_indices = []
    with multiprocessing.Pool(20) as pool:
        for sample_id, sample in enumerate(tqdm(pool.imap(Get_Window_Features, windows), total=len(windows))):

            if sample is not None:
                samples.append(sample)
            else:
                missing_indices.append(sample_id)

    # Create an array initialized with NaN
    solar_wind_probability = np.full(len(windows), np.nan)

    if samples: # Check if we have any samples
        samples = pd.concat(samples, ignore_index=True)
        valid_probabilities = random_forest.predict_proba(samples)[:, 1]
        valid_indices = [i for i in range(len(windows)) if i not in missing_indices]
        solar_wind_probability[valid_indices] = valid_probabilities

    else:
        raise ValueError("All samples missing")

    fig, (ax, probability_ax) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    ax.plot(data["date"], data["|B|"], color="black")

    colours = {"Solar Wind": "cornflowerblue", "Magnetosheath": "indianred"}

    uncertainty = True
    uncertainty_size = 0.2
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
    for i in range(len(window_centres) - 1):

        if i == 0:
            continue

        if uncertainty:
            if solar_wind[i]:
                ax.axvspan(
                    window_centres[i] - (window_centres[i] - window_centres[i - 1]) / 2,
                    window_centres[i] + (window_centres[i + 1] - window_centres[i]) / 2,
                    color="cornflowerblue",
                    edgecolor=None,
                    alpha=0.3,
                    label=sw_label,
                )
                sw_label = ""

            elif magnetosheath[i]:
                ax.axvspan(
                    window_centres[i] - (window_centres[i] - window_centres[i - 1]) / 2,
                    window_centres[i] + (window_centres[i + 1] - window_centres[i]) / 2,
                    color="indianred",
                    edgecolor=None,
                    alpha=0.3,
                    label=ms_label,
                )
                ms_label = ""

            else:
                ax.axvspan(
                    window_centres[i] - (window_centres[i] - window_centres[i - 1]) / 2,
                    window_centres[i] + (window_centres[i + 1] - window_centres[i]) / 2,
                    color="lightgrey",
                    edgecolor=None,
                    alpha=0.3,
                    label=uncertain_label,
                )
                uncertain_label = ""

        else:
            if solar_wind[i]:
                ax.axvspan(
                    window_centres[i] - (window_centres[i] - window_centres[i - 1]) / 2,
                    window_centres[i] + (window_centres[i + 1] - window_centres[i]) / 2,
                    color="cornflowerblue",
                    edgecolor=None,
                    alpha=0.3,
                    label=sw_label,
                )
                sw_label = ""

            else:
                ax.axvspan(
                    window_centres[i] - (window_centres[i] - window_centres[i - 1]) / 2,
                    window_centres[i] + (window_centres[i + 1] - window_centres[i]) / 2,
                    color="indianred",
                    edgecolor=None,
                    alpha=0.3,
                    label=ms_label,
                )
                ms_label = ""

    probability_ax.plot(
        window_centres, solar_wind_probability, color="blue", zorder=10, alpha=0.5, lw=2
    )

    ax.legend()

    ax.set_ylabel("|B| [nT]")
    probability_ax.set_ylabel("Solar Wind Probability")

    ax.set_title(
        f"Random Forest Application (Overlapping Sliding Window)\nWindow Size: {window_size} s    Step Size: {step_size} s"
    )

    ax.margins(0)
    probability_ax.margins(0)
    fig.subplots_adjust(hspace=0)
    plotting.Add_Tick_Ephemeris(ax)

    probability_ax.set_ylim(0, 1)
    probability_ax.axhline(0.5, color="grey", ls="dashed")

    # Add boundary crossing intervals
    boundaries.Plot_Crossing_Intervals(ax, start, end, crossings, color="black")

    plt.show()
