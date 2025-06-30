import numpy as np

start_counter = 0  
start_pos     = 0
start_speed   = 13.8 # [m/s] = ca. 50 km/h


max_simu_time = 60 # unit: seconds
dt = 1 # unit: seconds

start_pos = 0 # unit: meters
start_speed = 13.8 # unit: m/s = ca. 50 km/h

start_P = np.array([[10.0, 0.0 ],
                    [0.0,  1.0 ]])

process_noise_mean = np.array( [0.0, 0.0, 0.0] )

measurement_noise_mean = np.array( [0.0, 0.0, 0.0] )

measurement_each_nth_step = 2


# set process noise covariance matrix
process_noise_cov = np.array( [[0.01, 0.0,  0.0],
                               [0.0,  0.1,  0.0],
                               [0.0,  0.0,  1.0]
                              ])

# set measurement noise covariance matrix
measurement_noise_cov = np.array( [[0.01, 0.0, 0.0],
                                   [0.0,  2.0, 0.0],
                                   [0.0,  0.0, 2.0]
                                  ])