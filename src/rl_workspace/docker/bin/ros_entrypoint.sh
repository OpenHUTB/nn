#!/bin/bash
set -e

# setup ros environment
source "/catkin_ws/devel/setup.bash"
exec "$@"
