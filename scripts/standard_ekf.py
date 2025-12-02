import numpy as np

class EKF_Standard:
    """
    EKF implementation without the reiliability functions

    """

    def __init__(
        self,
        Q,
        R,
        alpha1, beta1, X_lidar_reli, Y_lidar_reli,
        alpha2, beta2, X_radar_reli, Y_radar_reli,
    ):
        # Process noise covariance Q (4x4, Eq. 10)
        self.Q = np.array(Q, dtype=float).reshape(4, 4)

        # Measurement noise covariance R_j (5x5, Eq. 11)
        self.R = np.array(R, dtype=float).reshape(5, 5)

        # Reliability parameters 
        self.alpha1 = float(alpha1)
        self.beta1  = float(beta1)
        self.X_lidar_reli = float(X_lidar_reli)
        self.Y_lidar_reli = float(Y_lidar_reli)

        self.alpha2 = float(alpha2)
        self.beta2  = float(beta2)
        self.X_radar_reli = float(X_radar_reli)
        self.Y_radar_reli = float(Y_radar_reli)

        # EKF state and covariance (initialized later)
        self.phi = None  
        self.P = None

    # EKF sigmoid reliability functions & derivatives (Eq. 8) 

    def set_initial_state(self, x0, P0=None):
        """
        Set initial state and covariance
        """
        self.phi = np.array(x0, dtype=float).reshape(4,)
        if P0 is None:
            self.P = np.eye(4, dtype=float)
        else:
            self.P = np.array(P0, dtype=float).reshape(4, 4)

    # EKF process model f and A (Eq. 4 & 5) 

    def f(self, x, dt):
        """
        Nonlinear process model (Eq. 4), applied from t-1 to t:
        """
        x_pos, y_pos, V, theta = x
        x_new = np.empty_like(x, dtype=float)
        x_new[0] = x_pos + dt * V * np.cos(theta)
        x_new[1] = y_pos + dt * V * np.sin(theta)
        x_new[2] = V
        x_new[3] = theta
        return x_new

    def A(self, x, dt):
        """
        Jacobian A_{t-1}^j (Eq. 5) 
        """
        _, _, V, theta = x
        A = np.eye(4, dtype=float)
        A[0, 2] = dt * np.cos(theta)
        A[1, 2] = dt * np.sin(theta)
        return A

    # EKF measurement model h and H (Eq. 6 & 7) 

        # EKF measurement model h and H (standard, no reliability) 

    def h(self, x):
        """
        Standard measurement function:
        z = [x_lidar, y_lidar, x_radar, y_radar, theta]^T
        where LiDAR and RADAR both directly observe (x, y) in ego frame,
        and theta is a direct heading measurement.
        """
        x_pos, y_pos, V, theta = x

        z_pred = np.array([
            x_pos,  # x_lidar
            y_pos,  # y_lidar
            x_pos,  # x_radar
            y_pos,  # y_radar
            theta   # theta
        ], dtype=float)

        return z_pred


    def H(self, x, z, dt):
        """
        Standard measurement Jacobian H
        
        """
        H = np.zeros((5, 4), dtype=float)

        # x_lidar = x
        H[0, 0] = 1.0

        # y_lidar = y
        H[1, 1] = 1.0

        # x_radar = x
        H[2, 0] = 1.0

        # y_radar = y
        H[3, 1] = 1.0

        # theta_meas = theta
        H[4, 3] = 1.0

        return H



    def predict(self, dt):
        """
        Prediction step (Eqs. 9 & 10):
        """
        A = self.A(self.phi, dt)
        self.phi = self.f(self.phi, dt)
        self.P = A @ self.P @ A.T + self.Q

    def update(self, z, dt):
        """
        Update step (Eqs. 11–13):

        """
        z = np.asarray(z, dtype=float).reshape(5,)

        # (Eq. 7)
        H = self.H(self.phi, z, dt)

        # Predicted measurement from current predicted state (Eq. 6)
        z_hat = self.h(self.phi)

        # Innovation
        y = z - z_hat

        # Innovation covariance
        S = H @ self.P @ H.T + self.R

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        self.phi = self.phi + K @ y

        # Covariance update
        I = np.eye(4, dtype=float)
        self.P = (I - K @ H) @ self.P