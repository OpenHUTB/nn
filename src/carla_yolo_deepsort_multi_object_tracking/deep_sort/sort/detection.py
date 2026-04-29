# vim: expandtab:ts=4:sw=4
import numpy as np


class Detection(object):
    """
    This class is uses to detect bounding box in a frame.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    """

    def __init__(self, tlwh, confidence, feature):
        self.tlwh = np.asarray(tlwh, dtype=np.float64)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)

     
    def convert_to_tlbr(self):

        """
        Convert bounding box to format (x,y,x,y) , i.e., (top left, bottom right).
        
        """
        min_x, min_y, width, height = self.tlwh
        max_x = min_x + width
        max_y = min_y + height
        return (min_x, min_y, max_x, max_y)

    def convert_to_xyah(self):

        """
        Convert bounding box to format (x_center, y_center, width/height, height).

        """
        min_x, min_y, width, height = self.tlwh
        x_center = min_x + width / 2
        y_center = min_y + height / 2
        a_ratio = width / height
        return (x_center, y_center, a_ratio, height)
    
