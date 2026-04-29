# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg

chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter(object):
    """
    This class implements simple Kalman filter to track bounding boxes in image space

    There are  8-dimensional state space

        x, y : Bounding box center position
        a : aspect ration (width/height)
        h : height 
        vx, vy : velocity in x and y direction
        va : angulat velocity
        vh : velocity in height change

     The bounding box location (x, y, a, h) is taken as direct observation of the state space.

    """

    def __init__(self):

        ndim, dt = 4, 1.

        # Create Kalman filter model matrix.

        self._motion_matrix = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_matrix[i, ndim + i] = dt
        self._update_matrix = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """ this function generate a track from unmatched detections

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        
        std_position = self._std_weight_position * measurement[3]
        std_velocities = self._std_weight_position * measurement[3]
        
        #creating the covariance matrix
        std = [ 2* std_position , 2 * std_position , 1e-2,
                2* std_position , 10 * std_velocities , 10 * std_velocities,
                  1e-5, 10 * std_velocities]
        
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """this function is the prediction step of kalman fileter

        Parameters
        ----------
        mean : ndarray , 8 dimensional vector
        covariance : ndarray , The 8x8 dimensional covariance matrix 
           
        """
        
        std_position = self._std_weight_position * mean[3]
        std_orientation = 1e-1
        std_velocities = self._std_weight_position * mean[3]
        
        #predicting the motion change of the detection
        
        std_mat = [std_position , std_position ,std_orientation ,std_velocities]
        std_vel = [std_position , std_position ,1e-5 , std_position]
          
        motion_cov = np.diag(np.square(np.r_[std_mat, std_vel]))

        mean = np.dot(self._motion_matrix, mean)
        covariance = np.linalg.multi_dot((
            self._motion_matrix, covariance, self._motion_matrix.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray , The state's mean vector (8 dimensional array).
        covariance : ndarray ,The state's covariance matrix (8x8 dimensional).
        
        """
        
        std_position = self._std_weight_position * mean[3]
        std_orientation = 1e-1
        std_velocities = self._std_weight_position * mean[3]
        
        std = [std_position, std_position, std_orientation, std_velocities]
        
        # Create the innovation covariance matrix
        innovation_cov = np.diag(np.square(std))
        
        # Perform the state projection
        mean = np.dot(self._update_matrix, mean)
        covariance = np.linalg.multi_dot((self._update_matrix, covariance, self._update_matrix.T))
        
        # Add innovation covariance to the updated covariance
        updated_covariance = covariance + innovation_cov
        
        return mean, updated_covariance

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray , predict 8 dimensional predicted mean vector
        covariance : ndarray , (8x8) covariance matrix
        measurement : ndarray , 4 dimension measurement vector (x, y, a, h), where (x, y)
            
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        
        #calculate kalman gain using Cholesky solver
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_matrix.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        
        #update the state covariance using kalman gain and predicted covariance
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Parameters
        ----------
        mean : ndarray , 8D Mean vector over the state distribution 
        covariance : ndarray , 8D Covariance of the state distribution 
        measurements : ndarray , An Nx4 dimensional matrix of N measurements 
        only_position : Optional[bool]
         
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
