import glob
import os
import tempfile
import carla
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pygame
import tqdm
from absl import logging

def setup(
    width: int = 400,
    height: int = 300,
    render: bool = True,
):
    """Returns the `display`, `clock` and for a `PyGame` app.

    Args:
        width: The width (in pixels) of the app window.
        height: The height (in pixels) of the app window.
        render: If True it renders a window, it keeps the
        frame buffer on the memory otherwise.

    Returns:
        display: The main app window or frame buffer object.
        clock: The main app clock.
        font: The font object used for generating text.
    """
    # PyGame setup.
    pygame.init()  # pylint: disable=no-member
    pygame.display.set_caption("OATomobile")
    if render:
        logging.debug("PyGame initializes a window display")
        display = pygame.display.set_mode(  # pylint: disable=no-member
            (width, height),
            pygame.HWSURFACE | pygame.DOUBLEBUF,  # pylint: disable=no-member
        )
    else:
        logging.debug("PyGame initializes a headless display")
        display = pygame.Surface((width, height))  # pylint: disable=too-many-function-args
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("dejavusansmono", 14)
    return display, clock, font

def make_dashboard(display, font, clock, observations) -> None:
    """Generates the dashboard used for visualizing the agent.

    Args:
        display: The `PyGame` renderable surface.
        observations: The aggregated observation object.
        font: The font object used for generating text.
        clock: The PyGame (client) clock.
    """
    # Clear dashboard.
    display.fill(COLORS["BLACK"])

    # Adaptive width.
    ada_width = 0

    if "preview_camera" in observations:
        # Render front camera view.
        ob_preview_camera_rgb = ndarray_to_pygame_surface(
            array=observations.get("preview_camera"),
            swapaxes=True,
        )
        display.blit(ob_preview_camera_rgb, (ada_width, 0))
        ada_width = ada_width + ob_preview_camera_rgb.get_width()

def ndarray_to_pygame_surface(array, swapaxes):

    """Returns a `PyGame` surface from a `NumPy` array (image).

    Args:
        array: The `NumPy` representation of the image to be converted to `PyGame`.

    Returns:
        A `PyGame` surface.
    """
    # Make sure its in 0-255 range.
    array = 255 * (array / array.max())
    if swapaxes:
        array = array.swapaxes(0, 1)
    return pygame.surfarray.make_surface(array)


# Color palette, the RGB values found at https://brandpalettes.com/.
COLORS = {
    # Default palette.
    "WHITE": pygame.Color(255, 255, 255),
    "BLACK": pygame.Color(0, 0, 0),
    "RED": pygame.Color(255, 0, 0),
    "GREEN": pygame.Color(0, 255, 0),
    "BLUE": pygame.Color(0, 0, 255),
    "SILVER": pygame.Color(195, 195, 195),
    # Google palette.
    "GOOGLE BLUE": pygame.Color(66, 133, 244),
    "GOOGLE RED": pygame.Color(219, 68, 55),
    "GOOGLE YELLOW": pygame.Color(244, 160, 0),
    "GOOGLE GREEN": pygame.Color(15, 157, 88),
    # Apple palettte.
    "APPLE MIDNIGHT GREEN": pygame.Color(78, 88, 81),
    "APPLE SPACE GREY": pygame.Color(83, 81, 80),
    "APPLE ROSE GOLD": pygame.Color(250, 215, 189),
    "APPLE LIGHT PURPLE": pygame.Color(209, 205, 218),
    "APPLE LIGHT YELLOW": pygame.Color(255, 230, 129),
    "APPLE LIGHT GREEN": pygame.Color(255, 230, 129),
    "APPLE SILVER": pygame.Color(163, 170, 174),
    "APPLE BLACK": pygame.Color(31, 32, 32),
    "APPLE WHITE": pygame.Color(249, 246, 239),
    "APPLE RED": pygame.Color(165, 40, 44),
    "APPLE GOLD": pygame.Color(245, 221, 197),
    # Slack palette.
    "SLACK AUBERGINE": pygame.Color(74, 21, 75),
    "SLACK BLUE": pygame.Color(54, 197, 240),
    # Other palettes.
    "AMAZON ORANGE": pygame.Color(255, 153, 0),
    "FACEBOOK BLUE": pygame.Color(66, 103, 178),
    "AIRBNB CORAL": pygame.Color(255, 88, 93),
    "DR.PEPPER MAROON": pygame.Color(113, 31, 37),
}