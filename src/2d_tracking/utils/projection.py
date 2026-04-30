import numpy as np

# Constants for min/max initialization
MAX_INIT_VALUE = float('inf')
MIN_INIT_VALUE = float('-inf')


def build_projection_matrix(w: int, h: int, fov: float,
                           is_behind_camera: bool = False) -> np.ndarray:
    """Build camera projection matrix from camera parameters.

    Args:
        w: Image width in pixels.
        h: Image height in pixels.
        fov: Field of view in degrees.
        is_behind_camera: Whether to flip the focal length sign.

    Returns:
        3x3 camera intrinsic matrix K.
    """
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)

    if is_behind_camera:
        K[0, 0] = K[1, 1] = -focal
    else:
        K[0, 0] = K[1, 1] = focal

    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def get_image_point(loc, K: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    """Calculate 2D projection of 3D coordinate.

    Args:
        loc: 3D location object with x, y, z attributes (e.g., carla.Position).
        K: Camera intrinsic matrix (3x3).
        w2c: World to camera transformation matrix (4x4).

    Returns:
        2D image point as [u, v, 1].
    """
    # Format the input coordinate
    point = np.array([loc.x, loc.y, loc.z, 1])
    # Transform to camera coordinates
    point_camera = np.dot(w2c, point)

    # Change from UE4's coordinate system to standard
    # (x, y, z) -> (y, -z, x)
    # Remove the fourth component
    point_camera = np.array(
        [point_camera[1], -point_camera[2], point_camera[0]]).T

    # Project 3D -> 2D using the camera matrix
    point_img = np.dot(K, point_camera)

    # Normalize
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img


def point_in_canvas(pos: list[float], img_h: int, img_w: int) -> bool:
    """Check if point is within image canvas boundaries.

    Args:
        pos: Point coordinates as [x, y].
        img_h: Image height in pixels.
        img_w: Image width in pixels.

    Returns:
        True if point is inside canvas, False otherwise.
    """
    return (0 <= pos[0] < img_w) and (0 <= pos[1] < img_h)


def get_vanishing_point(p1: list[float], p2: list[float],
                       p3: list[float], p4: list[float]) -> list[float]:
    """Calculate vanishing point from two pairs of parallel lines.

    Args:
        p1: First point of first line [x, y].
        p2: Second point of first line [x, y].
        p3: First point of second line [x, y].
        p4: Second point of second line [x, y].

    Returns:
        Vanishing point as [x, y].
    """
    k1 = (p4[1] - p3[1]) / (p4[0] - p3[0])
    k2 = (p2[1] - p1[1]) / (p2[0] - p1[0])

    vp_x = (k1 * p3[0] - k2 * p1[0] + p1[1] - p3[1]) / (k1 - k2)
    vp_y = k1 * (vp_x - p3[0]) + p3[1]

    return [vp_x, vp_y]


def get_2d_box_from_3d_edges(points_2d: np.ndarray, edges: list[list[int]],
                            image_h: int, image_w: int) -> tuple[float, float, float, float]:
    """Calculate 2D bounding box from 3D edges projection.

    Args:
        points_2d: Array of 2D points (N, 2).
        edges: List of edge pairs [[i, j], ...].
        image_h: Image height in pixels.
        image_w: Image width in pixels.

    Returns:
        Tuple of (x_min, x_max, y_min, y_max).
    """
    x_min, x_max = MAX_INIT_VALUE, MIN_INIT_VALUE
    y_min, y_max = MAX_INIT_VALUE, MIN_INIT_VALUE

    for edge in edges:
        p1 = points_2d[edge[0]]
        p2 = points_2d[edge[1]]

        p1_in_canvas = point_in_canvas(p1, image_h, image_w)
        p2_in_canvas = point_in_canvas(p2, image_h, image_w)

        # Both points are out of the canvas
        if not p1_in_canvas and not p2_in_canvas:
            continue

        # Draw 2D Bounding Boxes
        p1_temp, p2_temp = (p1.copy(), p2.copy())

        # One of the point is out of the canvas
        if not (p1_in_canvas and p2_in_canvas):
            p = [0, 0]

            # Find the intersection of the edge with the window border
            p_in_canvas, p_not_in_canvas = (
                p1, p2) if p1_in_canvas else (p2, p1)
            k = (p_not_in_canvas[1] - p_in_canvas[1]) / (p_not_in_canvas[0] - p_in_canvas[0])

            x = np.clip(p_not_in_canvas[0], 0, image_w)
            y = k * (x - p_in_canvas[0]) + p_in_canvas[1]

            if y >= image_h:
                p[0] = (image_h - p_in_canvas[1]) / k + p_in_canvas[0]
                p[1] = image_h - 1
            elif y <= 0:
                p[0] = (0 - p_in_canvas[1]) / k + p_in_canvas[0]
                p[1] = 0
            else:
                p[0] = image_w - 1 if x == image_w else 0
                p[1] = y

            p1_temp, p2_temp = (p, p_in_canvas)

        # Update bounding box coordinates
        x_max = max(x_max, p1_temp[0], p2_temp[0])
        x_min = min(x_min, p1_temp[0], p2_temp[0])
        y_max = max(y_max, p1_temp[1], p2_temp[1])
        y_min = min(y_min, p1_temp[1], p2_temp[1])

    return x_min, x_max, y_min, y_max
