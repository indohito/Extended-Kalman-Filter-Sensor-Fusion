import numpy as np

class EKF_Paper:
    """
    EKF implementation following Kim & Park,
    'Extended Kalman Filter (EKF) Design for Vehicle Position Tracking
    Using Reliability Function of Radar and Lidar', Sensors 2020.

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

    def _sig(self, x, alpha, beta, x0):
        """
        Reliability function (Eq. 8):
        """
        return beta / (1.0 + np.exp(alpha * (x - x0)))

    def _sig_prime(self, x, alpha, beta, x0):
        """
        (used inside Eq. 7).
        """
        g = 1.0 / (1.0 + np.exp(alpha * (x - x0)))
        return -alpha * beta * g * (1.0 - g)

    # EKF initialization 

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

    def h(self, x):
        """
        Measurement function corresponding to Eq. (6).
        """
        x_pos, y_pos, V, theta = x

        # lidar reliability terms
        dx_l = self._sig(x_pos, self.alpha1, self.beta1, self.X_lidar_reli)
        dy_l = self._sig(y_pos, self.alpha1, self.beta1, self.Y_lidar_reli)

        # radar reliability terms
        dx_r = self._sig(x_pos, -self.alpha2, self.beta2, self.X_radar_reli)
        dy_r = self._sig(y_pos, -self.alpha2, self.beta2, self.Y_radar_reli)

        z_pred = np.array([
            x_pos + dx_l,   # x_lidar
            y_pos + dy_l,   # y_lidar
            x_pos + dx_r,   # x_radar
            y_pos + dy_r,   # y_radar
            theta           # theta
        ], dtype=float)
        return z_pred

    def H(self, x, z, dt):
        """
        Measurement Jacobian H_t^j to match Equation (7)

        """
        x_pos, y_pos, V, theta = x
        x_lid, y_lid, x_rad, y_rad, theta_meas = z

        # radar x (uses -alpha2, beta2, X_radar_reli)
        d_sig_radar_x = self._sig_prime(
            x_rad, -self.alpha2, self.beta2, self.X_radar_reli
        )
        # lidar y (uses alpha1, beta1, Y_lidar_reli)
        d_sig_lidar_y = self._sig_prime(
            y_lid, self.alpha1, self.beta1, self.Y_lidar_reli
        )
        # radar y (uses -alpha2, beta2, Y_radar_reli)
        d_sig_radar_y = self._sig_prime(
            y_rad, -self.alpha2, self.beta2, self.Y_radar_reli
        )

        H = np.zeros((5, 4), dtype=float)

        # Row 1: x_lidar
        H[0, 0] = 1.0 - d_sig_radar_x
        H[0, 2] = dt * np.cos(theta)

        # Row 2: y_lidar
        H[1, 1] = 1.0 + d_sig_lidar_y
        H[1, 2] = dt * np.sin(theta)

        # Row 3: x_radar
        H[2, 0] = 1.0 - d_sig_radar_x
        H[2, 2] = dt * np.cos(theta)

        # Row 4: y_radar
        H[3, 1] = 1.0 - d_sig_radar_y
        H[3, 2] = dt * np.sin(theta)

        # Row 5: θ (heading)
        H[4, 3] = 1.0

        return H

    # EKF prediction & update (Eqs. 9–13) 

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