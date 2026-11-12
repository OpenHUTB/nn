"""Entry point: ``python main.py <command>``.

Commands: collect / train / eval / benchmark / demo / slam / teleop.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from mujoco_car_nn.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
