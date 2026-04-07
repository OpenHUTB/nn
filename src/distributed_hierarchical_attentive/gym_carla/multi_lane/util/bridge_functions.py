import numpy as np
import cv2
import sys
import utm
import math


def lidar_string_to_array(lidar, half_cloud=None, whole_cloud=None):
    """
    Return the LiDAR pointcloud in numpy.array format based on a string. Every time, half (in this case) of the cloud
    is computed due to the LiDAR frequency, so if whole_cloud == True, we concatenate two consecutive pointclouds
    """
    lidar_data = np.fromstring(lidar, dtype=np.float32)
    lidar_data = np.reshape(lidar_data, (int(lidar_data.shape[0] / 4), 4))

    # We take the oposite of y axis (since in CARLA a LiDAR point is
    # expressed in left-handed coordinate system, and ROS needs right-handed)

    lidar_data[:, 1] *= -1

    if whole_cloud:
        lidar_data = np.concatenate((half_cloud, lidar_data), axis=0)

    return lidar_data


def cv2_to_imgmsg(cvim, encoding="passthrough"):
    """
    Convert an OpenCV :cpp:type:`cv::Mat` type to a ROS sensor_msgs::Image message.
    :param cvim:      An OpenCV :cpp:type:`cv::Mat`
    :param encoding:  The encoding of the image data, one of the following strings:
        * ``"passthrough"``
        * one of the standard strings in sensor_msgs/image_encodings.h
    :rtype:           A sensor_msgs.msg.Image message
    :raises CvBridgeError: when the ``cvim`` has a type that is incompatible with ``encoding``
    If encoding is ``"passthrough"``, then the message has the same encoding as the image's OpenCV type.
    Otherwise desired_encoding must be one of the standard image encodings
    This function returns a sensor_msgs::Image message on success, or raises :exc:`cv_bridge.CvBridgeError` on failure.
    """

    if not isinstance(cvim, (np.ndarray, np.generic)):
        raise TypeError('Your input type is not a numpy array')
    img_msg = Image()
    img_msg.height = cvim.shape[0]
    img_msg.width = cvim.shape[1]

    if len(cvim.shape) < 3:
        cv_type = 'mono8'
    else:
        cv_type = 'bgr8'
    if encoding == "passthrough":
        img_msg.encoding = cv_type
    else:
        img_msg.encoding = encoding

    if cvim.dtype.byteorder == '>':
        img_msg.is_bigendian = True
    img_msg.data = cvim.tostring()
    img_msg.step = len(img_msg.data) // img_msg.height

    return img_msg


def build_camera_info(width, height, f_x, f_y, x, y, current_ros_time, frame_id, distorted_image=None):
    """
    Private function to compute camera info
    camera info doesn't change over time
    """
    camera_info = CameraInfo()
    camera_info.header.stamp = current_ros_time
    camera_info.header.frame_id = frame_id
    camera_info.width = width
    camera_info.height = height
    camera_info.distortion_model = 'plumb_bob'
    cx = camera_info.width / 2.0
    cy = camera_info.height / 2.0
    fx = f_x
    fy = f_y
    camera_info.K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]

    if not distorted_image:
        camera_info.D = [0, 0, 0, 0, 0]
        camera_info.R = [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]
        camera_info.P = [fx, 0, cx, x, 0, fy, cy, y, 0, 0, 1.0, 0]

        return camera_info
    else:
        return np.array([camera_info.K]).reshape(3, 3)  # Only return intrinsic parameters


def build_camera_info_from_file(frame, x_pos, y_pos, current_ros_time,
                                camera_parameters_path='/workspace/team_code/generic_modules/camera_parameters/'):
    """
    Private function to compute camera info
    camera info doesn't change over time
    """

    x = x_pos
    y = y_pos

    K = np.loadtxt(camera_parameters_path + 'K.txt')
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # print("x, y: ", x, y)
    # print("fx, fy, cx, cy: ", fx, fy, cx, cy)

    roi = np.loadtxt(camera_parameters_path + 'roi.txt')
    xtl, ytl, width, height = roi
    width = int(width)
    height = int(height)

    camera_info = CameraInfo()
    camera_info.header.stamp = current_ros_time
    camera_info.header.frame_id = frame
    camera_info.width = width
    camera_info.height = height
    camera_info.distortion_model = 'plumb_bob'

    camera_info.K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
    camera_info.D = [0, 0, 0, 0, 0]
    camera_info.R = [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]
    camera_info.P = [fx, 0, cx, x, 0, fy, cy, y, 0, 0, 1.0, 0]

    return camera_info


def image_rectification(distorted_image,
                        camera_parameters_path='/workspace/team_code/generic_modules/camera_parameters/'):
    """
    """

    K_distorted = np.loadtxt(camera_parameters_path + 'K_original.txt')  # Load your original K matrix
    D = np.loadtxt(camera_parameters_path + 'D.txt')  # Load the distortion coefficients of your original image
    roi = np.loadtxt(camera_parameters_path + 'roi.txt').astype(np.int64)  # Load ROI dimensions

    h_dist, w_dist = distorted_image.shape[:2]

    x, y, w, h = roi

    dst = cv2.undistort(distorted_image, K_distorted, D, None)  # Undistort
    dst = dst[y:y + h, x:x + w]
    return dst


def get_routeNodes(route):
    """
    Returns the route in Node3D format to visualize it on RVIZ
    """

    nodes = []

    for waypoint in route:
        node = monitor_classes.Node3D()
        node.x = waypoint.transform.location.x
        node.y = -waypoint.transform.location.y
        node.z = 0
        nodes.append(node)
    return nodes


def process_localization(ekf, gnss, imu, actual_speed, current_ros_time, map_frame, base_link_frame, enabled_pose,
                         count_localization, init_flag, last_yaw):
    """
    Return UTM position (x,y,z) and orientation of the ego-vehicle as a nav_msgs.Odometry ROS message based on the
    gnss information (WGS84) and imu (to compute the orientation)
        GNSS    ->  latitude =  gnss[0] ; longitude = gnss[1] ; altitude = gnss[2]
        IMU     ->  accelerometer.x = imu[0] ; accelerometer.y = imu[1] ; accelerometer.z = imu[2] ;
                    gyroscope.x = imu[3]  ;  gyroscope.y = imu[4]  ;  gyroscope.z = imu[5]  ;  compass = imu[6]
    """

    # Read GNSS data
    latitude = -gnss[0]  # Negative y to correspond to carla axis
    longitude = gnss[1]
    altitude = gnss[2]

    # Convert Geographic (latitude, longitude) to UTM (x,y) coordinates
    EARTH_RADIUS_EQUA = 6378137.0  # pylint: disable=invalid-name
    scale = math.cos(latitude * math.pi / 180.0)
    x = scale * longitude * math.pi * EARTH_RADIUS_EQUA / 180.0
    # Negative y to correspond to carla documentations
    y = - scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + latitude) * math.pi / 360.0))

    # Read IMU data -> Yaw angle is used to give orientation to the gnss pose
    roll = 0
    pitch = 0
    compass = imu[6]
    yaw_velocity = -imu[5]  ##Carla tiene los ejes de coordenadas detodo (mapa, sensores...) con la Y en sentido opuesto

    if np.isnan(compass):  # Sometimes we receive NaN measurement by compass
        yaw = last_yaw
        print("ERROR NaN received by IMU")
    else:
        if (0 < compass < math.radians(180)):
            yaw = -compass + math.radians(90)
        else:
            yaw = -compass + math.radians(450)
        last_yaw = yaw
    [qx, qy, qz, qw] = euler_to_quaternion(roll, pitch, yaw)

    # Process EKF filter
    if init_flag:
        ekf = Localization_EKF(np.array([x, y, yaw]))
    x_filtered, y_filtered = ekf.kalman_filter(x, y, actual_speed, yaw_velocity)

    # Filling in ROS messages for publishing
    gnss_pose_msg = Odometry()
    gnss_pose_msg.header.frame_id = map_frame
    gnss_pose_msg.child_frame_id = base_link_frame
    gnss_pose_msg.header.stamp = current_ros_time
    gnss_pose_msg.pose.pose.position.x = x
    gnss_pose_msg.pose.pose.position.y = y
    gnss_pose_msg.pose.pose.position.z = 0
    gnss_pose_msg.pose.pose.orientation.x = qx
    gnss_pose_msg.pose.pose.orientation.y = qy
    gnss_pose_msg.pose.pose.orientation.z = qz
    gnss_pose_msg.pose.pose.orientation.w = qw
    gnss_pose_msg.twist.twist.linear.x = actual_speed
    gnss_pose_msg.twist.twist.angular.z = yaw_velocity
    gnss_translation_error = 0.55  # std. deviation -> error_lat = error_long = 0.000005deg -> x_error = y_error = 0.55m
    gnss_rotation_error = 0.0  # error compass = 0 deg
    gnss_pose_msg.pose.covariance = np.diag(
        [gnss_translation_error, gnss_translation_error, 0.0, 0.0, 0.0, gnss_rotation_error]).ravel()
    speedometer_error = 0.0
    imu_gyroscope_error = 0.001  # standard deviation of yaw rate in rad/s
    gnss_pose_msg.twist.covariance = np.diag([speedometer_error, 0.0, 0.0, 0.0, 0.0, imu_gyroscope_error]).ravel()

    filtered_pose_msg = Odometry()
    filtered_pose_msg.header.frame_id = map_frame
    filtered_pose_msg.child_frame_id = base_link_frame
    filtered_pose_msg.header.stamp = current_ros_time
    filtered_pose_msg.pose.pose.position.x = x_filtered
    filtered_pose_msg.pose.pose.position.y = y_filtered
    filtered_pose_msg.pose.pose.position.z = 0
    filtered_pose_msg.pose.pose.orientation.x = qx
    filtered_pose_msg.pose.pose.orientation.y = qy
    filtered_pose_msg.pose.pose.orientation.z = qz
    filtered_pose_msg.pose.pose.orientation.w = qw
    filtered_pose_msg.twist.twist.linear.x = actual_speed

    if not enabled_pose:  ##Espera de 1seg converger EKF
        gnss_translation_error = 0.01  # [m]
        count_localization += 1
        if (count_localization >= 20):
            enabled_pose = True

    return ekf, filtered_pose_msg, gnss_pose_msg, enabled_pose, count_localization, last_yaw


class Localization_EKF():
    """
    """

    def __init__(self, initial_obs):
        """
            initial_obs: numpy array (x,y,yaw) w.r.t. map
        """
        xy_obs_noise_std = 0.556597453966366  # standard deviation of observation noise of x and y in meter
        initial_yaw_std = 0.0  # standard deviation of observation noise of yaw in radian
        forward_velocity_noise_std = 0.0
        yaw_rate_noise_std = 0.001  # standard deviation of yaw rate in rad/s

        self.P = np.array([
            [xy_obs_noise_std ** 2., 0., 0.],
            [0., xy_obs_noise_std ** 2., 0.],
            [0., 0., initial_yaw_std ** 2.]])
        self.Q = np.array([
            [xy_obs_noise_std ** 2., 0.],
            [0., xy_obs_noise_std ** 2.]])
        self.R = np.array([
            [forward_velocity_noise_std ** 2., 0., 0.],
            [0., forward_velocity_noise_std ** 2., 0.],
            [0., 0., yaw_rate_noise_std ** 2.]])

        self.kf = EKF(initial_obs, self.P)
        self.dt = 0.05

    def kalman_filter(self, x, y, lineal_velocity_x, ang_velocity_yaw):
        u = np.array([lineal_velocity_x, ang_velocity_yaw])
        self.kf.propagate(u, self.dt, self.R)
        z = np.array([x, y])
        self.kf.update(z, self.Q)
        return self.kf.x[0], self.kf.x[1]

