import sys

import moveit_commander
import rospy
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rospy_message_converter import message_converter

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node("move_group_python_interface", anonymous=True)


@app.get("/state")
async def state():
    robot = moveit_commander.RobotCommander()
    state = robot.get_current_state()
    serialized = message_converter.convert_ros_message_to_dictionary(state)
    return {"robot": serialized, "status": "ok"}


@app.get("/reset")
async def reset():
    return {"status": "ok"}


class Quaternion(BaseModel):
    x: float
    y: float
    z: float
    w: float


class Point(BaseModel):
    x: float
    y: float
    z: float


class Pose(BaseModel):
    position: Point
    orientation: Quaternion


@app.post("/controller/trajectory")
async def controller_trajectory(pose: Pose):
    group = moveit_commander.MoveGroupCommander("arm")
    group.set_pose_target(pose)
    group.go()
    group.stop()
    group.clear_pose_targets()
    return {"status": "ok"}


@app.get("/controller/trajectory_random")
async def controller_trajectory_random():
    group = moveit_commander.MoveGroupCommander("arm")
    group.set_random_target()
    group.go()
    group.stop()
    group.clear_pose_targets()
    return {"status": "ok"}


@app.post("/controller/{motor}/position")
async def controller_position(motor: str):
    return {"status": "ok"}


app.mount("/", StaticFiles(directory="../static", html=True), name="static")
