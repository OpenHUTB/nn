import cv2
import torch
import torch.nn.functional as F
import numpy as np
import threading
from queue import Queue
from train_segmentation import LitUnsupervisedSegmenter
from utils import get_transform
from crf import dense_crf
from os.path import join
from PIL import Image

# Paths and model loading
dir = "logs/checkpoints/stego_test_9/directory_carla_custom_1_clusterWarmup_L2_nclasses20_date_Jul15_14-27-25/"
sav_model = "epoch=24-step=14399.ckpt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the PyTorch model
model = LitUnsupervisedSegmenter.load_from_checkpoint(join(dir, sav_model)).to(device).eval().half()

vid_path = "../../testing_videos/"
video = "vid_town10HD_small_crash.mp4"
video_path = join(vid_path, video)

# Frame queue for real-time processing
frame_queue = Queue(maxsize=100)
segmented_frame_queue = Queue(maxsize=100)

# Batch size for processing frames
BATCH_SIZE = 4
resize_res = 448

def process_batch_with_stego(frames, model, use_linear_probe=True):
    original_height, original_width = frames[0].shape[:2]

    # Resize and transform frames for segmentation
    resized_frames = [cv2.resize(frame, (resize_res, resize_res), interpolation=cv2.INTER_NEAREST) for frame in frames]
    transform = get_transform(resize_res, False, "center")
    imgs = torch.stack([transform(Image.fromarray(resized_frame)).to(device).half() for resized_frame in resized_frames])

    with torch.no_grad():
        codes = model(imgs)  
        codes_flipped = model(imgs.flip(dims=[3]))
        codes = (codes + codes_flipped.flip(dims=[3])) / 2
        codes = F.interpolate(codes, imgs.shape[-2:], mode='bilinear', align_corners=False)
        
        segmented_frames = []
        for img, code in zip(imgs, codes):
            if use_linear_probe:
                linear_probs = torch.log_softmax(model.linear_probe(code.unsqueeze(0)), dim=1).cpu()
                seg_pred = dense_crf(img.cpu().float(), linear_probs[0].float()).argmax(0)
            else:
                cluster_probs = model.cluster_probe(code.unsqueeze(0), 2, log_probs=True).cpu()
                seg_pred = dense_crf(img.cpu().float(), cluster_probs[0].float()).argmax(0)

            segmented_frame = model.label_cmap[seg_pred].astype(np.uint8)
            segmented_frame = cv2.cvtColor(segmented_frame, cv2.COLOR_BGR2RGB)
            segmented_frame = cv2.resize(segmented_frame, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
            segmented_frames.append(segmented_frame)

    return segmented_frames

def read_frames():
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            frame_queue.put(None)
            break
        frame_queue.put(frame)
    cap.release()

def process_frames():
    while True:
        batch_frames = []
        while len(batch_frames) < BATCH_SIZE:
            frame = frame_queue.get()
            if frame is None:
                break
            batch_frames.append(frame)
        if not batch_frames:
            segmented_frame_queue.put(None)
            break
        segmented_frames = process_batch_with_stego(batch_frames, model)
        for original_frame, segmented_frame in zip(batch_frames, segmented_frames):
            segmented_frame_queue.put((original_frame, segmented_frame))

def display_segmented_frames():
    while True:
        frame_pair = segmented_frame_queue.get()
        if frame_pair is None:
            break
        original_frame, segmented_frame = frame_pair
        combined_frame = np.hstack((original_frame, segmented_frame)).astype(np.uint8)
        cv2.imshow('Segmented Frame', combined_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

reader_thread = threading.Thread(target=read_frames)
processor_thread = threading.Thread(target=process_frames)
display_thread = threading.Thread(target=display_segmented_frames)

reader_thread.start()
processor_thread.start()
display_thread.start()

reader_thread.join()
processor_thread.join()
display_thread.join()

print("Real-time video segmentation completed.")
