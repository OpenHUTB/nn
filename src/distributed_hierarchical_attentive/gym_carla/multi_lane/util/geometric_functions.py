import numpy as np

def euler_to_quaternion(roll, pitch, yaw):
	"""
    Return the orientation of our ego-vehicle in quaternion based on Euler angles (Roll = x, Pitch = y, Yaw = z)
	"""
	qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
	qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
	qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
	qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

	return [qx, qy, qz, qw]   

def unit_vector(vector):
    """ 
    Return the corresponding unit vector
    """
    return vector / np.linalg.norm(vector)