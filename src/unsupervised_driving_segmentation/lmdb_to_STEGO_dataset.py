from pathlib import Path

import torch
import lmdb
import gc
import psutil
import os
import glob
import numpy as np
import cv2
import tqdm
import numpy as np
from PIL import Image
from torchvision import transforms as T
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import argparse
import yaml

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from CBS2.cbs2.bird_view.utils.datasets.birdview_lmdb import seg2D_to_ND, Transform, Rotation, Location
import math
import random
from CBS2.cbs2.bird_view.utils.image_utils import CoordinateConverter

from PIL import Image
import torchvision.transforms as T

#import augmenter
from CBS2.cbs2.bird_view.augmenter_wor import augment

PIXEL_OFFSET = 10
PIXELS_PER_METER = 5
SEG_CLASSES = {4, 6, 7, 10, 18}  # pedestrians, roadlines, roads, vehicles, tl
width = 0
height = 0

def world_to_pixel(x,y,ox,oy,ori_ox, ori_oy, offset=(-80,160), size=320, angle_jitter=15):
    pixel_dx, pixel_dy = (x-ox)*PIXELS_PER_METER, (y-oy)*PIXELS_PER_METER

    pixel_x = pixel_dx*ori_ox+pixel_dy*ori_oy
    pixel_y = -pixel_dx*ori_oy+pixel_dy*ori_ox

    pixel_x = size-pixel_x

    return np.array([pixel_x, pixel_y]) + offset


def project_to_image(pixel_x, pixel_y, tran=[0.,0.,0.], rot=[0.,0.,0.], fov=90, w=width, h=height, camera_world_z=1.4, crop_size=192):
    # Apply fixed offset tp pixel_y
    pixel_y -= 2*PIXELS_PER_METER

    pixel_y = crop_size - pixel_y
    pixel_x = pixel_x - crop_size/2

    world_x = pixel_x / PIXELS_PER_METER
    world_y = pixel_y / PIXELS_PER_METER

    xyz = np.zeros((1,3))
    xyz[0,0] = world_x
    xyz[0,1] = camera_world_z
    xyz[0,2] = world_y

    f = w /(2 * np.tan(fov * np.pi / 360))
    A = np.array([
        [f, 0., w/2],
        [0, f, h/2],
        [0., 0., 1.]
    ])
    image_xy, _ = cv2.projectPoints(xyz, np.array(tran), np.array(rot), A, None)
    image_xy[...,0] = np.clip(image_xy[...,0], 0, w)
    image_xy[...,1] = np.clip(image_xy[...,1], 0, h)

    return image_xy[0,0]

class ImageDataset(Dataset):
    def __init__(self,
        dataset_path,
        rgb_shape=(height,width,3),
        img_size=320,
        crop_size=192,
        gap=5,
        n_step=5,
        gaussian_radius=1.,
        down_ratio=4,
        augment_strategy=None,
        batch_read_number=819200,
        batch_aug=1,
        buffer=40,
        combine_seg=False,
):
        self._name_map = {}

        self.file_map = {}
        self.idx_map = {}

        self.bird_view_transform = transforms.ToTensor()
        self.rgb_transform = transforms.ToTensor()

        self.rgb_shape = rgb_shape
        self.img_size = img_size
        self.crop_size = crop_size

        self.gap = gap
        self.ori_gap = gap
        self.buffer = buffer
        self.combine_seg = combine_seg

        self.n_step = n_step
        self.down_ratio = down_ratio
        self.batch_aug = batch_aug

        self.gaussian_radius = gaussian_radius

        # CBS RGB image augmentation
        # print("augment with ", augment_strategy)
        # if augment_strategy is not None and augment_strategy != 'None':
        #     self.augmenter = getattr(augmenter, augment_strategy)
        # else:
        #     self.augmenter = None

        # For CBS2, we use WoR RGB image augmentation
        print(f'WoR data augmentation approach, with batch_aug of {self.batch_aug} and p= 0.5')
        # self.augmenter = augment(0.5)
        self.augmenter = None

        count = 0
        for full_path in glob.glob('%s/**'%dataset_path):
            lmdb_file = lmdb.open(full_path,
                 max_readers=1,
                 readonly=True,
                 lock=False,
                 readahead=False,
                 meminit=False
            )

            txn = lmdb_file.begin(write=False)

            dataset_length = int(txn.get('len'.encode()))
            N = dataset_length - (self.gap + self.buffer) * self.n_step
            
            # Ensure N is non-negative
            if N < 0:
                N = 0
            for _ in range(N):
                self._name_map[_+count] = full_path
                self.file_map[_+count] = txn
                self.idx_map[_+count] = _

            count += N

        print("Finished loading %s. Length: %d"%(dataset_path, count))
        self.batch_read_number = batch_read_number

    def __len__(self):
        return len(self.file_map)

    def project_vehicle(self, x, y, z, ori_x, ori_y, ori_z):
        pos = np.array([x, y, z])
        #ori = np.array([ori_x, ori_y, ori_z])
        #ori /= np.linalg.norm(ori)  # Make unit vector

        #new_pos = pos + 4 * ori
        fwd_2d_angle = np.deg2rad(ori_y) #yaw to rad
        new_pos = pos + 5.5 * np.array([np.cos(fwd_2d_angle), np.sin(fwd_2d_angle), 0])
        new_pos_cam_coords = self.converter.convert(np.array([new_pos]))
        if(new_pos_cam_coords.shape[0] == 0):
            return np.array([[192, 147, 0]]) # In the center of the image, almost at the bottom --> stop waypoint
        return new_pos_cam_coords

    @staticmethod
    def interpolate_waypoints(points):
        points = points[:, :2]

        # Fit first or second function through points
        n_degree = 2 if points.shape[0] > 2 else 1
        z = np.polyfit(points[:, 0], points[:, 1], n_degree)
        p = np.poly1d(z)

        # Keep interpolating until we have n_step=5 points
        while points.shape[0] < 5:
            points_2 = np.vstack([points[0], points[:-1]])
            max_id = np.argmax(np.linalg.norm(points - points_2, axis=1))
            _x = np.mean([points[max_id], points_2[max_id]], axis=0)[0]
            points = np.insert(points, max_id, np.array([_x, p(_x)]), 0)

        return points

    def get_waypoints(self, index, lmdb_txn, world_x, world_y, world_z, ori_x, ori_y, ori_z):
        tl = int.from_bytes(lmdb_txn.get(('trafficlights_%04d' % index).encode()), 'little')
        speed = np.frombuffer(lmdb_txn.get(('spd_%04d'%index).encode()), np.float32)[0]

        output = []
        #if tl or vehicle or walker:
        if speed < 0.005:
            vehicle_proj = self.project_vehicle(world_x, world_y, world_z, ori_x, ori_y, ori_z)
            output = np.array([vehicle_proj[0] for _ in range(self.n_step)])
            if tl:
                return output, 3 # Stop TL
            else:
                return output, 4 # Stop obstacle


        for i in range(index, (index + (self.n_step + 1 + self.buffer * self.gap)), self.gap):
            if len(output) == self.n_step:
                break

            x, y, z = np.frombuffer(lmdb_txn.get(('loc_%04d' % i).encode()), np.float32)
            image_coords = self.converter.convert(np.array([[x, y, z]]))
            if len(image_coords) > 0:
                output.append(image_coords[0])

        if len(output) < 2:
            # First try with smaller GAP
            if self.gap == self.ori_gap:
                self.gap = 1
                return self.get_waypoints(index, lmdb_txn, world_x, world_y, world_z,ori_x, ori_y, ori_z)

            vehicle_proj = self.project_vehicle(world_x, world_y, world_z, ori_x,ori_y, ori_z)
            output = np.array([vehicle_proj[0] for _ in range(self.n_step)])
            return output, 2 # Less than two waypoints --> stop

        if 2 <= len(output) < self.n_step:
            return self.interpolate_waypoints(np.array(output)), 1 # Interpolation

        return np.array(output), 0 # All waypoints ok

    @staticmethod
    def down_scale(img):
        new_shape = (img.shape[0] // 2, img.shape[1] // 2)
        img = np.moveaxis(img, 1, 0)
        img = cv2.resize(img.astype(np.float32), new_shape)
        img = np.moveaxis(img, 1, 0)

        return img

    def __getitem__(self, idx):
        if idx not in self.file_map:
            raise KeyError(f"Index {idx} not found in file_map")

        lmdb_txn = self.file_map[idx]
        index = self.idx_map[idx]

        # bird_view = np.frombuffer(lmdb_txn.get(('birdview_%04d'%index).encode()), np.uint8).reshape(320,320,7)
        seg_buffer = np.frombuffer(lmdb_txn.get(('segmentation_%04d'%index).encode()), np.uint8)
        segmentation = seg_buffer.reshape(height, width, 3)
        col_seg = map_labels_to_colors(segmentation)
        segmentation = self.down_scale(segmentation)
        
        assert_shape = (height/2, width/2, 3)
        assert segmentation.shape == assert_shape, "Incorrect shape ({}), got {}".format(assert_shape, segmentation.shape)

        tl_info = int.from_bytes(lmdb_txn.get(('trafficlights_%04d' % index).encode()), 'little')

        segmentation = seg2D_to_ND(segmentation, tl_info).astype(np.float32)

        ox, oy, oz = np.frombuffer(lmdb_txn.get(('loc_%04d'%index).encode()), np.float32)
        ori_ox, ori_oy, ori_oz = np.frombuffer(lmdb_txn.get(('rot_%04d'%index).encode()), np.float32)
        speed = np.frombuffer(lmdb_txn.get(('spd_%04d'%index).encode()), np.float32)[0]
        cmd = int(np.frombuffer(lmdb_txn.get(('cmd_%04d'%index).encode()), np.float32)[0])+1 # 1:Left, 2:Right, 3:Straight, 4:Follow
        cam_x, cam_y, cam_z = np.frombuffer(lmdb_txn.get(('cam_location_%04d'%index).encode()), np.float32)
        cam_pitch, cam_yaw, cam_roll = np.frombuffer(lmdb_txn.get(('cam_rotation_%04d'%index).encode()), np.float32)

        rgb_image = np.frombuffer(lmdb_txn.get(('rgb_%04d'%index).encode()), np.uint8).reshape(height,width,3)

        if self.augmenter:
            rgb_images = [self.augmenter(image=rgb_image) for i in range(self.batch_aug)]
        else:
            rgb_images = [rgb_image for i in range(self.batch_aug)]

        if self.batch_aug == 1:
            rgb_images = rgb_images[0]

        # Create coordinate transformer
        sensor_transform = Transform(Location(cam_x, cam_y, cam_z),
                                     Rotation(cam_pitch, cam_yaw, cam_roll))
        self.converter = CoordinateConverter(sensor_transform, fov=120)

        # Get waypoints in image coordinates (x, y)
        image_coord_wp, wp_method = self.get_waypoints(index, lmdb_txn, ox, oy, oz,ori_ox, ori_oy,ori_oz)
        image_coord_wp = image_coord_wp[:,:2].astype(np.float32)

        self.gap = self.ori_gap  # Reset gap to its original value

        # segmentation = segmentation.astype(np.float32)
        image_coord_wp = np.array(image_coord_wp, dtype=np.float32)

        # assert_shape = (height/2, width/2)
        # assert segmentation.shape == assert_shape, "Incorrect shape ({}), got {}".format(assert_shape, segmentation.shape)
        # assert len(image_coord_wp) == self.n_step, "Not enough points, got {}".format(image_coord_wp.shape)

        if self.batch_aug == 1:
            rgb_images = self.rgb_transform(rgb_images)
        else:
            rgb_images = torch.stack([self.rgb_transform(img) for img in rgb_images])

        # segmentation = self.bird_view_transform(segmentation)

        self.batch_read_number += 1

        return rgb_images, col_seg, segmentation, image_coord_wp, cmd, speed, wp_method


def load_image_data(dataset_path,
        batch_size=32,
        num_workers=8,
        shuffle=True,
        n_step=5,
        gap=10,
        augment=None,
        **kwargs
    ):

    dataset = ImageDataset(
        dataset_path,
        n_step=n_step,
        gap=gap,
        augment_strategy=augment,
        **kwargs,
    )

    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, drop_last=True, pin_memory=True)


class Wrap(Dataset):
    def __init__(self, data, batch_size, samples):
        self.data = data
        self.batch_size = batch_size
        self.samples = samples

    def __len__(self):
        return self.batch_size * self.samples

    def __getitem__(self, i):
        return self.data[np.random.randint(len(self.data))]


def _dataloader(data, batch_size, num_workers):
    return DataLoader(
            data, batch_size=batch_size, num_workers=num_workers,
            shuffle=True, drop_last=True, pin_memory=True)


def get_image(
        dataset_dir,
        batch_size=32, num_workers=0, shuffle=True, augment=None,
        n_step=5, gap=5, batch_aug=1):

    def make_dataset(dir_name, is_train):
        _dataset_dir = str(Path(dataset_dir) / dir_name)
        _samples = 1000 if is_train else 10
        _num_workers = num_workers if is_train else 0
        _batch_aug = batch_aug if is_train else 1
        _augment = augment if is_train else None

        data = ImageDataset(
                _dataset_dir, gap=gap, n_step=n_step, augment_strategy=_augment, batch_aug=_batch_aug)
        data = Wrap(data, batch_size, _samples)
        data = _dataloader(data, batch_size, _num_workers)

        return data 

    train = make_dataset('train', True)
    val = make_dataset('val', False)

    return train, val


# Define color mapping for CARLA segmentation labels
COLOR_MAP = {
    0: (0, 0, 0),          # Unlabeled
    1: (70, 70, 70),       # Building
    2: (100, 40, 40),      # Fence
    3: (55, 90, 80),       # Other
    4: (220, 20, 60),      # Pedestrian
    5: (153, 153, 153),    # Pole
    6: (157, 234, 50),     # Roadline
    7: (128, 64, 128),     # Road
    8: (244, 35, 232),     # Sidewalk
    9: (107, 142, 35),     # Vegetation
    10: (0, 0, 142),       # Vehicle
    11: (102, 102, 156),   # Wall
    12: (220, 220, 0),     # TrafficSign
    13: (70, 130, 180),    # Sky
    14: (81, 0, 81),       # Ground
    15: (150, 100, 100),   # Bridge
    16: (230, 150, 140),   # RailTrack
    17: (180, 165, 180),   # GuardRail
    18: (250, 170, 30),    # TrafficLight
    19: (110, 190, 160),   # Static
    20: (170, 120, 50),    # Dynamic
    21: (45, 60, 150),     # Water
    22: (145, 170, 100)    # Terrain
}

def map_labels_to_colors(segmentation, sem_colors=COLOR_MAP):
    sem = segmentation[:, :, 2]
    # Create an empty canvas with 3 channels
    canvas = np.zeros((sem.shape[0], sem.shape[1], 3), dtype=np.uint8)
    unique_labels = np.unique(sem)
    max_unilabels_per_image = len(unique_labels)
    
    for label in unique_labels:
        if label in sem_colors:
            mask = sem == label
            canvas[mask] = sem_colors[label]
        else:
            print(f"Warning: Label {label} not in COLOR_MAP, skipping.")
    return canvas

def extract_segmentation(lmdb_txn, index):
    # Retrieve the segmentation image
    segmentation = np.frombuffer(lmdb_txn.get(('segmentation_%04d' % index).encode()), np.uint8).reshape(height, width)
    return segmentation

def save_colored_segmentation(segmentation, output_path, index):
    colored_segmentation = map_labels_to_colors(segmentation)
    pil_image = Image.fromarray(colored_segmentation)
    pil_image.save(f'{output_path}/segmentation_frame_{index}.png')

def process_image(rgb_img, col_seg, j, i, output_path, data_subset):
    try:
        if isinstance(rgb_img, torch.Tensor):
            rgb_img = rgb_img.permute(1, 2, 0).cpu().numpy()
        if isinstance(col_seg, torch.Tensor):
            col_seg = col_seg.cpu().numpy()
        
        # Ensure data types are compatible with PIL
        rgb_img = (rgb_img* 255).astype(np.uint8)
        col_seg = col_seg.astype(np.uint8)

        pil_image = Image.fromarray(rgb_img)
        os.makedirs(f'{output_path}/imgs/{data_subset}', exist_ok=True)
        pil_image.save(f'{output_path}/imgs/{data_subset}/{data_subset}_{j}_{i}_bird_view_frame_RGB.jpg')
        # print(f"Saved image: {output_path}/imgs/{data_subset}/{data_subset}_{j}_{i}_bird_view_frame_RGB.jpg")

        pil_seg = Image.fromarray(col_seg)
        os.makedirs(f'{output_path}/labels/{data_subset}', exist_ok=True)
        pil_seg.save(f'{output_path}/labels/{data_subset}/lab_{data_subset}_{j}_{i}_bird_view_frame_SEG.png')
        # print(f"Saved segmentation: {output_path}/labels/{data_subset}/lab_{data_subset}_{j}_{i}_bird_view_frame_SEG.png")
    except Exception as e:
        print(f"Error processing image {j}_{i}: {e}")

def main(args):
    batch_size = args.batch_size
    output_path = args.output_path
    dataset_path = args.dataset_path
    dataset_size = args.dataset_size
    num_workers = args.num_workers
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    global width, height
    width = config['width']
    height = config['height']

    # Val chosen as initial directory is a critical decision. As this smaller in size compared to train. Thus the dataset size check will immediately check, instead of failing later on and losing precious time
    for data_subset in ('val', 'train'):
        dataset = ImageDataset(dataset_path + data_subset)

        if len(dataset) == 0:
            print(f"No data found in the dataset at {dataset_path + data_subset}. Please check the dataset path and contents.")
            return
        loader = _dataloader(dataset, batch_size=batch_size, num_workers=num_workers)
        # to_pil_image = T.ToPILImage()
        process = psutil.Process(os.getpid())

        # Usable variables: (bird_view, locations, cmd, speed, wp_method)
        for j, (rgb_imgs, col_segs, bird_view, locations, cmd, speed, wp_method) in enumerate(tqdm.tqdm(loader)):
            # Use Parallel processing for saving images
            if dataset_size == 'large':
                Parallel(n_jobs=cpu_count())(
                    delayed(process_image)(rgb_imgs[i], col_segs[i], j, i, output_path, data_subset) for i in range(rgb_imgs.size(0))
                )
            elif dataset_size == 'small':
                for i in range(rgb_imgs.size(0)):
                    process_image(rgb_imgs[i], col_segs[i], j, i, output_path, data_subset)
            elif dataset_size == 'test':
                try:
                    Parallel(n_jobs=cpu_count())(
                        delayed(process_image)(rgb_imgs[i], col_segs[i], j, i, output_path, data_subset) for i in range(rgb_imgs.size(0))
                    )
                except Exception as e:
                    print(f"Parallel processing failed with error: {e}. Falling back to normal processing.")
                    for i in range(rgb_imgs.size(0)):
                        process_image(rgb_imgs[i], col_segs[i], j, i, output_path, data_subset)

            print(f"Memory usage: {process.memory_info().rss / (1024 ** 2)} MB")
            gc.collect()  # Clear unused memory 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process Carla dataset to jpg\png images for STEGO training phase.')
    parser.add_argument('--batch_size', type=int, default=24, help='Batch size for processing')
    parser.add_argument('--output_path', type=str, default='/storage/felix/Afstudeerproject/final_512_256_stego/', help='Output path for saving images')
    parser.add_argument('--dataset_path', type=str, default='/storage/felix/Afstudeerproject/final_training_split512_256/', help='Path to the dataset')
    parser.add_argument('--dataset_size', choices=['small', 'large', 'test'], default='large', help='Size of dataset to determine whether parallel processing can be used')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers for data loading')
    parser.add_argument('--config_path', type=str, default='CBS2/autoagents/collector_agents/config_data_collection.yaml', help='Path to the config file containing resolution for dataset video/images')

    args = parser.parse_args()
    main(args)
