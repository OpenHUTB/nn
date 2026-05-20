import numpy as np

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

def rotationMatrix2D(theta):
   rad = np.radians(theta)
   c = np.cos(rad)
   s = np.sin(rad)
   return np.array([[c, -s], [s, c]])

def clip(value, min_value, max_value):
   if value < min_value:
      return min_value
   if value > max_value:
      return max_value
   return value