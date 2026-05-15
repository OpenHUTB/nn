import numpy as np

def bilinear_resize(image, height, width):
  """
  `image` is a 2-D numpy array
  `height` and `width` are the desired spatial dimension of the new 2-D array.
  """
  img_height, img_width = image.shape

  image = image.ravel()

  x_ratio = float(img_width - 1) / (width - 1) if width > 1 else 0
  y_ratio = float(img_height - 1) / (height - 1) if height > 1 else 0

  y, x = np.divmod(np.arange(height * width), width)

  x_l = np.floor(x_ratio * x).astype('int32')
  y_l = np.floor(y_ratio * y).astype('int32')

  x_h = np.ceil(x_ratio * x).astype('int32')
  y_h = np.ceil(y_ratio * y).astype('int32')

  x_weight = (x_ratio * x) - x_l
  y_weight = (y_ratio * y) - y_l

  a = image[y_l * img_width + x_l]
  b = image[y_l * img_width + x_h]
  c = image[y_h * img_width + x_l]
  d = image[y_h * img_width + x_h]

  resized = a * (1 - x_weight) * (1 - y_weight) + \
            b * x_weight * (1 - y_weight) + \
            c * y_weight * (1 - x_weight) + \
            d * x_weight * y_weight

  return resized.reshape(height, width)
