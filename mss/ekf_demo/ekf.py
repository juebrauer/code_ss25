import numpy as np

class ekf:

    def __init__(self, inital_x,
                       inital_P,
                       est_f,
                       est_h,
                       Jacobi_f,
                       Jacobi_h,
                       Q,
                       R):
        self.x = inital_x 
        self.P = inital_P
        self.est_f = est_f
        self.est_h = est_h
        self.Jacobi_f = Jacobi_f
        self.Jacobi_h = Jacobi_h      
        self.Q = Q
        self.R = R

        self.n = len(self.x)

    def predict(self):

        F = self.Jacobi_f(self.x)
        self.P = F @ self.P @ F.T + self.Q

        self.x = self.est_f( self.x )
       


    def measurement_correction(self, z):
        
        self.y = z - self.est_h( self.x )

        H = self.Jacobi_h( self.x )

        self.S = H @ self.P @ H.T + self.R
        self.K = self.P @ H.T @ np.linalg.inv(self.S)

        self.x = self.x + self.K @ self.y
        self.P = (np.eye(self.n) - self.K @ H) @ self.P


    def get_scalar_measure_of_uncertainty_about_state(self):
        return np.linalg.det( self.P )