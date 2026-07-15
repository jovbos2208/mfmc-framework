import numpy as np
import pymsis as ps
import pandas as pd

column_names = ["MASS_DENSITY", "N2", "O2", "O", "HE", "H", "AR", "N", "AO", "NO", "TEMP"]

dates = np.arange(np.datetime64("2006-01-01T00:00"),
                  np.datetime64("2007-01-01T00:00"),
                  np.timedelta64(480, "m"))

lats = np.linspace(-90, 90, 13)
longs = np.linspace(-180, 180, 13)
heights = np.linspace(200, 500, 7)

# Constant orbital speed (m/s) by altitude (km). Replace if you want.
# Circular-orbit approximation: v = sqrt(mu / r)
MU = 3.986004418e14  # m^3/s^2
RE = 6378137.0       # m
v_by_h = {int(h): float(np.sqrt(MU / (RE + h * 1000.0))) for h in heights}

data = ps.calculate(dates, longs, lats, heights)

for k, h in enumerate(heights):
    df = pd.DataFrame(columns=column_names)
    for i in range(13):
        for j in range(13):
            new_df = pd.DataFrame(data[:, i, j, k, :], columns=column_names)
            df = pd.concat([df, new_df], ignore_index=True)

    v = v_by_h[int(h)]  # m/s
    df["DYN_PRESSURE"] = 0.5 * df["MASS_DENSITY"].astype(float) * (v ** 2)  # Pa

    df.to_csv(f"atmos_data/database_{int(h):03d}km.csv", index=False)
    print(f'Downloading data for alt: {int(h):03d}km done!')
