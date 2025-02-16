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
from skimage.restoration import denoise_tv_chambolle
import scipy.stats
from tqdm import tqdm
from hermpy import mag, utils, trajectory, boundaries, plotting


colours = ["black", "#DC267F", "#648FFF", "#FFB000"]

crossings = boundaries.Load_Crossings(utils.User.CROSSING_LISTS["Philpott"])
crossings = crossings.loc[crossings["Type"].str.contains("BS")]
crossings = crossings.loc[
    crossings["Start Time"].between(dt.datetime(2011, 3, 1), dt.datetime(2011, 12, 31))
]

# Shuffle crossings
crossings = crossings.sample(frac=1)

crossings = crossings.loc[crossings["Start Time"].between(dt.datetime(2011, 4, 11, 11), dt.datetime(2011, 4, 12))]

time_buffer = dt.timedelta(minutes=10)

# Import application parameters
window_size = 10  # seconds. How large of a window to feed to the random forest
step_size = 1  # seconds. How far should the window jump each time

smoothing = "TVD"  # "TVD", "BoxCar", "None"
smoothing_size = 5

remove_smallest_regions = False
region_length_minimum = 10  # times step size

skip_low_success = True


# Load Model
print("Loading model")
with open(
    "/home/daraghhollman/Main/Work/mercury/DataSets/bow_shock_gradient_boosting", "rb"
) as file:
    random_forest: sklearn.ensemble.HistGradientBoostingClassifier = pickle.load(file)

for i, crossing in crossings.iterrows():
    print(f"Processing crossing {i}")
    # Import time series
    start = crossing["Start Time"] - time_buffer
    end = crossing["End Time"] + time_buffer

    print(f"Loading data between {start} and {end}")
    data = mag.Load_Between_Dates(
        utils.User.DATA_DIRECTORIES["MAG_FULL"], start, end, average=None
    )

    windows = [
        (window_start, window_start + dt.timedelta(seconds=window_size))
        for window_start in pd.date_range(start=start, end=end, freq=f"{step_size}s")
    ]
    window_centres = [
        window_start + (window_end - window_start) / 2
        for window_start, window_end in windows
    ]

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
    with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as pool:
        for sample_id, sample in enumerate(
            tqdm(pool.imap(Get_Window_Features, windows), total=len(windows))
        ):

            if sample is not None:
                samples.append(sample)
            else:
                missing_indices.append(sample_id)

    # Create an array initialized with NaN
    solar_wind_probability = np.full(len(windows), np.nan)

    if samples:  # Check if we have any samples
        samples = pd.concat(samples, ignore_index=True)
        valid_probabilities = random_forest.predict_proba(samples)[:, 1]
        valid_indices = [i for i in range(len(windows)) if i not in missing_indices]
        solar_wind_probability[valid_indices] = valid_probabilities

    else:
        raise ValueError("All samples missing")

    # Smoothing
    match smoothing:
        case "TVD":
            solar_wind_probability = denoise_tv_chambolle(solar_wind_probability)

        case "BoxCar":
            solar_wind_probability = pd.Series(solar_wind_probability)
            solar_wind_probability = solar_wind_probability.rolling(
                window=smoothing_size
            ).median()

        case "None":
            pass

        case _:
            raise ValueError("Unknown choice of smoothing method.")

    fig, (ax, probability_ax) = plt.subplots(
        2, 1, gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    ax.plot(
        data["date"],
        data["|B|"],
        color=colours[0],
        lw=1,
    )

    uncertainty = False
    uncertainty_size = 0.1
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
        uncertainty_size = 0
        solar_wind = solar_wind_probability > 0.5
        magnetosheath = solar_wind_probability < 0.5

        sw_label = f"P(SW) > {0.5}"
        ms_label = f"P(SW) < {0.5}"

    # Determine success rate
    # We calculate a proxy for how accurate the application is by looking at
    # the ratio of samples *outside* of the boundary crossing interval,
    # which disagree with the region expectation.
    sw_predictions_before_bci = solar_wind_probability[
        np.array(window_centres) < crossing["Start Time"]
    ]
    sw_predictions_after_bci = solar_wind_probability[
        np.array(window_centres) > crossing["End Time"]
    ]

    # Use the bow shock direction to determine which region is before and after
    if crossing["Type"] == "BS_IN":
        # Get the proportion of misclassifcations to total
        # We expect sw before BS_IN, i.e. P(SW) > 0.5
        success_rate_before = sum(sw_predictions_before_bci > 0.5) / len(
            sw_predictions_before_bci
        )
        success_rate_after = sum(sw_predictions_after_bci < 0.5) / len(
            sw_predictions_after_bci
        )
        total_success_rate = (success_rate_before + success_rate_after) / 2

    elif crossing["Type"] == "BS_OUT":
        # Get the proportion of misclassifcations to total
        # We expect ms before BS_OUT, i.e. P(SW) < 0.5
        success_rate_before = sum(sw_predictions_before_bci < 0.5) / len(
            sw_predictions_before_bci
        )
        success_rate_after = sum(sw_predictions_after_bci > 0.5) / len(
            sw_predictions_after_bci
        )
        total_success_rate = (success_rate_before + success_rate_after) / 2

    else:
        raise ValueError("Crossing is not a bow shock")

    if total_success_rate < 0.65 and skip_low_success:
        print(f"Skipping due to low success rate: {total_success_rate}")
        continue

    print(f"Success Rate: {total_success_rate}")

    # Remove small chains
    if remove_smallest_regions:

        def Remove_Small_Chains(solar_wind_probability, region_length_minimum):
            # Initialize the first region
            chain_length = 1
            if solar_wind_probability[0] > 0.5:
                previous_region = "SW"
            else:
                previous_region = "MSh"

            new_predictions = []
            # Loop through predictions and find chain lengths
            for sw_prediction in solar_wind_probability[1:]:
                if sw_prediction > 0.5:
                    current_region = "SW"
                else:
                    current_region = "MSh"

                if current_region == previous_region:
                    chain_length += 1
                else:
                    # The chain has ended, check if it's too short
                    if chain_length < region_length_minimum:
                        # Flip the entire chain to be current region
                        new_predictions.extend([current_region] * chain_length)
                    else:
                        # Keep the current chain
                        new_predictions.extend([previous_region] * chain_length)

                    # Reset for the next chain
                    previous_region = current_region
                    chain_length = 1

            # Handle the last chain after the loop
            if chain_length < region_length_minimum:
                flipped_region = "MSh" if previous_region == "SW" else "SW"
                new_predictions.extend([flipped_region] * chain_length)
            else:
                new_predictions.extend([previous_region] * chain_length)

            return new_predictions

        # We need to run the removal code twice to fix edge cases such as:
        # [1, 0, 1, 1, 1, 1]
        new_predictions = Remove_Small_Chains(
            solar_wind_probability, region_length_minimum
        )
        solar_wind = np.array(new_predictions) == "SW"
        new_predictions = Remove_Small_Chains(
            solar_wind, region_length_minimum
        )
        solar_wind = np.array(new_predictions) == "SW"
        new_predictions = Remove_Small_Chains(
            solar_wind, region_length_minimum
        )

        # Convert final predictions to boolean arrays
        solar_wind = np.array(new_predictions) == "SW"
        magnetosheath = np.array(new_predictions) == "MSh"

    # Add region shading
    # Iterate through each window
    for i in range(len(window_centres) - 1):

        if i == 0:
            continue

        if uncertainty:
            if solar_wind[i]:
                ax.axvspan(
                    window_centres[i] - (dt.timedelta(seconds=step_size)) / 2,
                    window_centres[i] + (dt.timedelta(seconds=step_size)) / 2,
                    color="cornflowerblue",
                    edgecolor=None,
                    alpha=0.3,
                    label=sw_label,
                )
                sw_label = ""

            elif magnetosheath[i]:
                ax.axvspan(
                    window_centres[i] - (dt.timedelta(seconds=step_size)) / 2,
                    window_centres[i] + (dt.timedelta(seconds=step_size)) / 2,
                    color="indianred",
                    edgecolor=None,
                    alpha=0.3,
                    label=ms_label,
                )
                ms_label = ""

            else:
                ax.axvspan(
                    window_centres[i] - (dt.timedelta(seconds=step_size)) / 2,
                    window_centres[i] + (dt.timedelta(seconds=step_size)) / 2,
                    color="lightgrey",
                    edgecolor=None,
                    alpha=0.3,
                    label=uncertain_label,
                )
                uncertain_label = ""

        else:
            uncertain_label = ""
            if solar_wind[i]:
                ax.axvspan(
                    window_centres[i] - (dt.timedelta(seconds=step_size)) / 2,
                    window_centres[i] + (dt.timedelta(seconds=step_size)) / 2,
                    color="cornflowerblue",
                    edgecolor=None,
                    alpha=0.3,
                    label=sw_label,
                )
                sw_label = ""

            else:
                ax.axvspan(
                    window_centres[i] - (dt.timedelta(seconds=step_size)) / 2,
                    window_centres[i] + (dt.timedelta(seconds=step_size)) / 2,
                    color="indianred",
                    edgecolor=None,
                    alpha=0.3,
                    label=ms_label,
                )
                ms_label = ""

    probability_upper = np.ma.masked_where(
        solar_wind_probability <= 0.5 + uncertainty_size, solar_wind_probability
    )
    if uncertainty:
        probability_mid = np.ma.masked_where(
            (solar_wind_probability <= 0.5 - uncertainty_size)
            | (solar_wind_probability >= 0.5 + uncertainty_size),
            solar_wind_probability,
        )
    probability_lower = np.ma.masked_where(
        solar_wind_probability >= 0.5 - uncertainty_size, solar_wind_probability
    )

    probability_ax.plot(
        window_centres,
        solar_wind_probability,
        color="black",
        lw=2,
        ls="dotted",
        alpha=0.5,
    )

    probability_ax.plot(
        window_centres, probability_lower, color="indianred", label=ms_label, lw=3
    )

    if uncertainty:
        probability_ax.plot(
            window_centres,
            probability_mid,
            color="lightgrey",
            label=uncertain_label,
            lw=3,
        )

    probability_ax.plot(
        window_centres, probability_upper, color="cornflowerblue", label=sw_label, lw=3
    )

    ax.legend()

    ax.set_ylabel("|B| [nT]")
    probability_ax.set_ylabel("Solar Wind Probability")

    ax.set_title(
        f"Gradient Boosting Application (Overlapping Sliding Window)\nWindow Size: {window_size} s    Step Size: {step_size} s"
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
