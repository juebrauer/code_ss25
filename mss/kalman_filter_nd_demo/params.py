import numpy as np

max_simu_time = 60 # unit: seconds
dt = 1 # unit: seconds

start_pos = 0 # unit: meters
start_speed = 13.8 # unit: m/s = ca. 50 km/h

start_P = np.array([[10.0, 0.0 ],
                    [0.0,  1.0 ]])

# [new_pos, new_speed] = [pos+dt*speed, speed]
true_F = np.array( [[1.0, dt],
                    [0.0, 1.0]]
                  )

u = np.array( [0.003858025]) # unit: m/s^2
# e.g., car accelerates from 50 km/h to 100 km/h in our hour:
# a [m/s^2] = delta_speed / delta_time = 50000 / 3600s^2 = 0.0083858025

true_B = np.array( [[0.0],
                    [dt]] )

process_noise_mean = np.array( [0.0, 0.0] )

process_noise_cov = np.array( [[1.0, 0.0],
                               [0.0, 0.1]] )

true_H = np.array( [[1.1, 0.0],
                    [0.0, 0.9]])

measurement_noise_mean = np.array( [0.0, 0.0] )

measurement_noise_cov = np.array( [[10000.0, 0.0],
                                   [0.0, 0.1]] )