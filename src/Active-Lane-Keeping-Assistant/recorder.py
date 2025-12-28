import cv2
import numpy as np

from os.path import join

class Recorder():
    """The recorder records a run of a simulation and saves it as a video file
    """

    def __init__(self,
                fps:int=30,
                height:int=480,
                width:int=640,
                folder:str='videos') -> None:
        """Constructor for the recorder

        Args:
            fps (int, optional): Number of fps for video clip. Defaults to 15.
            height (int, optional): Height of video. Defaults to 480.
            width (int, optional): Width of video. Defaults to 640.
            folder (str, optional): Storage location. Defaults to 'videos'.
        """

        #self.cameraCapture = cv2.VideoCapture(0)
        self.height = height
        self.width = width
        self.fps = float(fps)
        self.folder = folder

        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video = None

    def init_new_video(self,
                    id:str) -> None:
        """Triggers the process of recording a new video 

        Args:
            id (str): Id/name of the video file
        """
        self.video = cv2.VideoWriter(join(self.folder, f'{id}.mp4'),
                                        self.fourcc,
                                        self.fps,
                                        (self.width, self.height))

    def add_image(self,
                img:np.ndarray) -> None:
        """Appends a new image to the video

        Args:
            img (np.ndarray): Frame to append to the video.
        """
        self.video.write(img)

    def close_recording(self) -> None:
        """Stops the recording

        Raises:
            Exception:
                If no recording has been initiated, the video can't be released.
        """
        if self.video is None:
            raise Exception('No video initialized.')
        self.video.release()

        def __enter__(self):
            """Context manager entry point"""
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            """Context manager exit point - ensures video is closed"""
            if self.video is not None:
                self.close_recording()