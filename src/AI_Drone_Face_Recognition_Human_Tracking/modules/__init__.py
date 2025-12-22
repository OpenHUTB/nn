# modules/__init__.py
"""
AI无人机人脸跟踪系统模块包
"""

from .drone_controller import DroneController
from .face_detector import FaceDetector
from .person_detector import PersonDetector
from .face_recognizer import FaceRecognizer
from .ui_controller import UIController
from .voice_synthesizer import VoiceSynthesizer

__all__ = [
    'DroneController',
    'FaceDetector',
    'PersonDetector',
    'FaceRecognizer',
    'UIController',
    'VoiceSynthesizer'
]