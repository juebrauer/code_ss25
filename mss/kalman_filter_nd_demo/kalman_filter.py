import numpy as np

class kalman_filter:

    def __init__(self, inital_x,
                       inital_P,
                       F,
                       B,
                       H,
                       Q,
                       R):
        self.x = inital_x 
        self.P = inital_P

        self.F = F
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R

        self.n = len(self.x)

    def predict(self, u):

        self.x = self.F @ self.x + self.B @ u
        self.P = self.F @ self.P @ self.F.T + self.Q


    def measurement_correction(self, z):
        
        self.y = z - (self.H @ self.x)
        self.S = self.H @ self.P @ self.H.T + self.R
        self.K = self.P @ self.H.T @ np.linalg.inv(self.S)

        self.x = self.x + self.K @ self.y
        self.P = (np.eye(self.n) - self.K @ self.H) @ self.P
