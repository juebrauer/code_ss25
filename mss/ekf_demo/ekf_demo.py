import matplotlib.pyplot as plt
import params
import numpy as np
import ekf

def f(x):
    """
    f: R^3 -> R^3
    f(x) = f(x0,x1,x2) = (x0+1, sin(x0), cos(x0))
                       = (f1(x), f2(x), f3(x))
    """
    new_counter = x[0] + 1
    new_pos     = np.sin(x[0])
    new_speed   = np.cos(x[0])
    new_x = np.array( [new_counter, new_pos, new_speed] )
    return new_x

def Jacobi_f(x):
    Jf = np.array([[            1, 0, 0],
                   [ np.cos(x[0]), 0, 0],
                   [-np.sin(x[0]), 0, 0]])
    return Jf

def h(x):
    """
    h(x) = h((x0,x1,x2)) = (x0**2, x1**2, x2**2) = (h1(x), h2(x), h3(x))
    """
    return x**2

def Jacobi_h(x):
    Jh = np.array([[ 2*x[0],      0,      0],
                   [      0, 2*x[1],      0],
                   [      0,      0, 2*x[2]]])
    return Jh


simu_time = 0.0
initial_x = np.array([params.start_pos, params.start_speed])
initial_P = params.start_P
true_x = initial_x.copy()
true_xs = []
zs = []
kf_ests = []
state_uncertainties = []

initial_x = np.array( [params.start_counter,
                       params.start_pos,
                       params.start_speed] )
initial_P = np.array( [[0.0,  0.0, 0.0],
                       [0.0 , 0.0, 0.0],
                       [0.0 , 0.0, 0.0]
                      ]
                    )

ekf = ekf.ekf(inital_x = initial_x,
              inital_P = initial_P,
              est_f = f,
              est_h = h,
              Jacobi_f = Jacobi_f,
              Jacobi_h = Jacobi_h, 
              Q = params.process_noise_cov,
              R = params.measurement_noise_cov)

while simu_time < params.max_simu_time:
    
    process_noise = np.random.multivariate_normal(params.process_noise_mean,
                                                  params.process_noise_cov)
    true_x = f(true_x) + process_noise


    measurement_noise = np.random.multivariate_normal(params.measurement_noise_mean,
                                                      params.measurement_noise_cov)
    z = h(true_x) + measurement_noise
    
    simu_time += params.dt

    ekf.predict()

    # Only do a measurement correction step each <measurement_each_nth_step>-th step
    if simu_time !=0 and simu_time % params.measurement_each_nth_step:
        ekf.measurement_correction(z)
    
    kf_est = ekf.x

    true_xs.append( true_x)
    zs.append( z )
    kf_ests.append( kf_est )

    uncertainty = ekf.get_scalar_measure_of_uncertainty_about_state()
    state_uncertainties.append( uncertainty )

plt.figure(figsize=(15,10))


# Mean Absolute Error
err_meas_pos  = np.mean([abs(true_x[1]-z[1])       for (true_x, z)      in zip(true_xs, zs)])
err_kf_pos    = np.mean([abs(true_x[1]-kf_est[1])  for (true_x, kf_est) in zip(true_xs, kf_ests)])
err_meas_speed = np.mean([abs(true_x[2]-z[2])      for (true_x, z)      in zip(true_xs, zs)])
err_kf_speed   = np.mean([abs(true_x[2]-kf_est[2]) for (true_x, kf_est) in zip(true_xs, kf_ests)])

plt.subplot(3,1, 1)
plt.title("Position")
plt.xlabel("time [s]")
plt.ylabel("position [m]")
plt.plot( [true_x[1] for true_x in true_xs], color="black", label="true" )
plt.plot( [z[1] for z in zs],                color="blue",  label=f"measured ({err_meas_pos:.1f})" )
plt.plot( [kf_est[1] for kf_est in kf_ests], color="red",   label=f"EKF estimate ({err_kf_pos:.1f})" )
plt.legend()

plt.subplot(3,1, 2)
plt.title("Speed")
plt.xlabel("time [s]")
plt.ylabel("speed [m/s]")
plt.plot( [true_x[2] for true_x in true_xs], color="black", label="true" )
plt.plot( [z[2] for z in zs],                color="blue",  label=f"measured ({err_meas_speed:.1f})" )
plt.plot( [kf_est[2] for kf_est in kf_ests], color="red",   label=f"EKF estimate ({err_kf_speed:.1f})" )
plt.legend()

plt.subplot(3,1, 3)
plt.title("State Uncertainty")
plt.xlabel("time [s]")
plt.ylabel("Uncertainty [det(CovMatrix)]")
plt.plot( state_uncertainties, color="black")

plt.subplots_adjust(hspace=1)
plt.show()



