import requests
import numpy as np
import cv2
import time
from collections import deque

from custom_faster_rcnn import CustomFasterRCNN

class FrameClient:
    """Client for fetching and displaying single-frame camera from server."""
    
    def __init__(self, url, retry_delay=2, fps_window_size=30):
        self.url = url
        self.retry_delay = retry_delay
        self.frame_times = deque(maxlen=fps_window_size)
        self.window_name = "Frame Stream"
        self.detector = CustomFasterRCNN(num_classes=4)

    def fetch_frame(self):
        """Fetch one JPEG frame from the server and decode it."""
        resp = requests.get(self.url, timeout=5)
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}")
        img_array = np.frombuffer(resp.content, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("Failed to decode frame")
        return frame

    def show_fps(self):
        """Display windowed FPS in the console."""
        if len(self.frame_times) > 1:
            avg_fps = len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0])
            print(f"\rWindowed FPS: {avg_fps:.1f}", end="")

    def run(self):
        """Main loop: fetch frames, display, track FPS, handle reconnects."""
        print(f"Connecting to {self.url} in FRAME mode. Press ESC to stop.")
        try:
            while True:
                try:

                    frame = self.fetch_frame()
                    self.frame_times.append(time.time())
                    self.show_fps()

                    boxes, labels, scores = self.detector.predict(frame)
                    frame = self.detector.draw_detections(frame, boxes, labels, scores)
                    cv2.imshow("Stream with Detections", frame)

                    # cv2.imshow(self.window_name, frame)
                    if cv2.waitKey(1) == 27:  # ESC
                        break

                except Exception:
                    print(f"\nStream not available. Retrying in {self.retry_delay}s...")
                    time.sleep(self.retry_delay)

        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    client = FrameClient("http://127.0.0.1:8080/frame")
    client.run()
