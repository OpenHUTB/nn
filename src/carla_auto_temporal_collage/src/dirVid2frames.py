import cv2
import os

def get_total_frames(video_path):
    """
    Get the total number of frames in the video.
    
    :param video_path: Path to the video file.
    :return: Total number of frames in the video.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

def extract_frames(video_path, output_dir, frame_interval=1):
    """
    Extract frames from a video file and save them as images.
    
    :param video_path: Path to the video file.
    :param output_dir: Directory where extracted frames will be saved.
    :param frame_interval: Interval of frames to save (default is every frame).
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get the total number of frames in the video
    total_frames = get_total_frames(video_path)
    
    # Determine the padding length based on the total number of frames
    padding_length = len(str(total_frames))
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Error opening video file: {video_path}")
    
    frame_count = 0
    saved_frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame at the specified interval
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{saved_frame_count:0{padding_length}d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"Saved {saved_frame_count} frames to '{output_dir}'")

def find_mp4_directories(data_dir):
    """
    Find all .mp4 files in the given directory and its subdirectories.
    
    :param data_dir: Root directory to search for .mp4 files.
    :return: List of paths to .mp4 files.
    """
    mp4_paths = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.mp4'):
                mp4_paths.append(os.path.join(root, file))
    return mp4_paths

def process_videos(data_dir, output_base_dir, frame_interval=1):
    """
    Process all .mp4 videos in the given directory and its subdirectories,
    extracting frames and saving them in a new directory structure.
    
    :param data_dir: Root directory containing subdirectories with .mp4 files.
    :param output_base_dir: Base directory where extracted frames will be saved.
    :param frame_interval: Interval of frames to save (default is every frame).
    """
    mp4_files = find_mp4_directories(data_dir)
    
    for video_path in mp4_files:
        # Derive the output directory path
        relative_path = os.path.relpath(video_path, data_dir)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(output_base_dir, os.path.dirname(relative_path), video_name)
        
        # Extract frames from the video
        extract_frames(video_path, output_dir, frame_interval)

# Example usage
data_dir = "data/videos"
output_base_dir = "data/data-frames/data-frames-3fps"
frame_interval = 10  # Save every 10th frame

process_videos(data_dir, output_base_dir, frame_interval)
