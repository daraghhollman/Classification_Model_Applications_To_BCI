"""
A script to test the loading of high res data from MESSENGER
"""

import datetime as dt
import matplotlib.pyplot as plt
from hermpy import mag, utils, boundaries

# Import time series
start = dt.datetime(2011, 8, 31, 0, 0)
end = dt.datetime(2011, 8, 31, 23, 59)
data = mag.Load_Between_Dates(utils.User.DATA_DIRECTORIES["MAG_FULL"], start, end, average=None)

mag.Remove_Spikes(data)

fig, ax = plt.subplots()

ax.plot(data["date"], data["|B|"], color="black")

crossings = boundaries.Load_Crossings(utils.User.CROSSING_LISTS["Philpott"])
boundaries.Plot_Crossing_Intervals(ax, start, end, crossings)

plt.show()
