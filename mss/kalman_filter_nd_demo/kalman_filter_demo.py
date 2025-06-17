import matplotlib.pyplot as plt
import params
import numpy as np
import kalman_filter


simu_time = 0.0
initial_x = np.array([params.start_pos, params.start_speed])
initial_P = params.start_P
true_x = initial_x.copy()
true_xs = []
zs = []
kf_ests = []

kf = kalman_filter.kalman_filter(inital_x = initial_x,
                                 inital_P = initial_P,
                                 F = params.true_F * 0.99,
                                 B = params.true_B,
                                 H = params.true_H,
                                 Q = params.process_noise_cov,
                                 R = params.measurement_noise_cov)

while simu_time < params.max_simu_time:
    
    process_noise = np.random.multivariate_normal(params.process_noise_mean,
                                                  params.process_noise_cov)
    true_x = params.true_F @ true_x + params.true_B @ params.u + process_noise


    measurement_noise = np.random.multivariate_normal(params.measurement_noise_mean,
                                                      params.measurement_noise_cov)
    z = params.true_H @ true_x + measurement_noise
    
    simu_time += params.dt

    kf.predict(params.u)
    kf.measurement_correction(z)
    kf_est = kf.x

    true_xs.append( true_x)
    zs.append( z )
    kf_ests.append( kf_est )

plt.figure(figsize=(15,10))

plt.subplot(2,1, 1)
plt.title("Position")
plt.xlabel("time [s]")
plt.ylabel("position [m]")
plt.plot( [true_x[0] for true_x in true_xs], color="black", label="true" )
plt.plot( [z[0] for z in zs],                color="blue",  label="measured" )
plt.plot( [kf_est[0] for kf_est in kf_ests], color="red",   label="KF estimate" )
plt.legend()

plt.subplot(2,1, 2)
plt.title("Speed")
plt.xlabel("time [s]")
plt.ylabel("speed [m/s]")
plt.plot( [true_x[1] for true_x in true_xs], color="black", label="true" )
plt.plot( [z[1] for z in zs],                color="blue",  label="measured" )
plt.plot( [kf_est[1] for kf_est in kf_ests], color="red",   label="KF estimate" )
plt.legend()

plt.show()



