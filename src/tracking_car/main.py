import argparse
import carla
import queue
import random
import cv2
import numpy as np
import time
import os
import sys
import yaml
import threading
import torch
import csv
import open3d as o3d
import psutil
from ultralytics import YOLO
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment
from datetime import datetime
from numba import njit
from loguru import logger
from sklearn.cluster import DBSCAN

# ======================== 通用工具函数 ========================
def valid_img(img): 
    return img is not None and len(img.shape)==3 and img.shape[2]==3 and img.size>0

def clip_box(bbox, img_shape):
    h,w = img_shape
    return np.array([
        max(0,min(bbox[0],w-1)),
        max(0,min(bbox[1],h-1)),
        max(bbox[0]+1,min(bbox[2],w-1)),
        max(bbox[1]+1,min(bbox[3],h-1))
    ], dtype=np.float32)

def make_div(x, d=32): 
    return (x + d -1) // d * d

# ======================== 常量配置 ========================
PLATFORM, IS_WIN, IS_LINUX = sys.platform, sys.platform.startswith('win'), sys.platform.startswith('linux')
WEATHER = {
    'clear':carla.WeatherParameters(0.0,0.0,0.0,0.0,180.0,75.0,0.0,0.0,1.0,0.0,0.0),
    'rain':carla.WeatherParameters(80.0,80.0,50.0,30.0,180.0,45.0,20.0,50.0,0.8,80.0,0.5),
    'fog':carla.WeatherParameters(90.0,0.0,0.0,10.0,180.0,30.0,70.0,20.0,0.5,10.0,0.8),
    'night':carla.WeatherParameters(20.0,0.0,0.0,0.0,0.0,-90.0,10.0,100.0,0.7,0.0,1.0),
    'cloudy':carla.WeatherParameters(90.0,0.0,0.0,20.0,180.0,60.0,10.0,100.0,0.9,0.0,0.3),
    'snow':carla.WeatherParameters(90.0,90.0,80.0,40.0,180.0,20.0,30.0,30.0,0.6,50.0,0.7)
}
VEHICLE_CLS = {2:"Car",5:"Bus",7:"Truck",-1:"Unknown"}

# ======================== 配置类（新增检测框样式配置） ========================
@dataclass
class Config:
    host, port, num_npcs = "localhost", 2000, 20
    img_width, img_height = 640, 480
    conf_thres, iou_thres, max_age, min_hits = 0.5, 0.3, 5, 3
    yolo_model, yolo_imgsz_max, yolo_iou, yolo_quantize = "yolov8n.pt", 320, 0.45, False
    kf_dt, max_speed = 0.05, 50.0
    window_width, window_height, smooth_alpha, fps_window_size, display_fps = 1280,720,0.2,15,30
    track_history_len, track_line_width, track_alpha = 20,2,0.6
    stop_speed_thresh, stop_frames_thresh = 1.0,5
    overtake_speed_ratio, overtake_dist_thresh = 1.5,50.0
    lane_change_thresh, brake_accel_thresh, turn_angle_thresh, danger_dist_thresh, predict_frames = 0.5,2.0,15.0,10.0,10
    default_weather, auto_adjust_detection = "clear", True
    use_lidar, lidar_channels, lidar_range, lidar_points_per_second, fuse_lidar_vision = True,32,100.0,500000,True
    record_data, record_dir, record_format, record_fps, save_screenshots = True,"track_records","csv",10,False
    use_3d_visualization, pcd_view_size = True,800
    
    # 新增：检测框样式配置（匹配截图）
    det_box_color: tuple = (255, 0, 0)  # 检测框颜色（蓝色）
    det_box_thickness: int = 2           # 框线宽度
    det_label_bg_color: tuple = (0, 0, 0)# 标签背景色（黑色）
    det_label_text_color: tuple = (255, 255, 255)  # 标签文字色（白色）
    det_label_font_scale: float = 0.5    # 字体大小
    det_label_padding: int = 5           # 标签内边距

    @classmethod
    def from_yaml(cls, p=None):
        try:
            p = p or os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
            if not os.path.exists(p): return cls()
            with open(p, "r", encoding="utf-8") as f: data = yaml.safe_load(f.read().strip().replace("\t","  "))
            if not isinstance(data, dict): return cls()
            valid_keys = set(cls.__dataclass_fields__.keys())
            data = {k:v for k,v in data.items() if k in valid_keys}
            for k,v in data.items():
                try: data[k] = cls.__dataclass_fields__[k].type(v)
                except: del data[k]
            return cls(**data)
        except: return cls()

# ======================== 天气图像增强 ========================
class WeatherEnhancer:
    def __init__(self, cfg):
        self.cfg = cfg; self.weather = "clear"
        self.params = {
            'clear':{'b':1.0,'c':1.0,'g':1.0},
            'rain':{'b':1.1,'c':1.2,'g':0.9,'dh':True,'dr':True},
            'fog':{'b':1.3,'c':1.4,'g':0.8,'dh':True},
            'night':{'b':1.5,'c':1.3,'g':0.7,'dn':True},
            'cloudy':{'b':1.2,'c':1.1,'g':1.0},
            'snow':{'b':1.1,'c':1.3,'g':0.9,'dh':True,'ds':True}
        }
    def set_weather(self, w):
        if w in WEATHER: self.weather = w
    def enhance(self, img):
        if not self.cfg.auto_adjust_detection or not valid_img(img): return img
        p = self.params.get(self.weather, self.params['clear'])
        enh = cv2.convertScaleAbs(img.copy(), alpha=p['c'], beta=int(p['b']*255-255))
        g = p['g']; inv_g = 1.0/g
        g_table = np.array([((i/255.0)**inv_g)*255 for i in range(256)]).astype(np.uint8)
        enh = cv2.LUT(enh, g_table)
        if p.get('dh'): enh = self._dehaze(enh)
        if p.get('dr'): enh = self._derain(enh)
        if p.get('ds'): enh = self._desnow(enh)
        if p.get('dn'): enh = cv2.fastNlMeansDenoisingColored(enh, None,10,10,7,21)
        return enh
    def _dehaze(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dc = cv2.erode(gray, np.ones((7,7), np.uint8))
        nz = dc[dc<10]; al = 255.0 if len(nz)==0 else np.max(img[dc<10])
        t = np.clip(1-0.1*(gray/al),0.1,1.0)
        deh = np.zeros_like(img, dtype=np.float32)
        for c in range(3): deh[:,:,c] = (img[:,:,c].astype(np.float32)-al)/t + al
        return np.clip(deh,0,255).astype(np.uint8)
    def _derain(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        km = cv2.getStructuringElement(cv2.MORPH_RECT, (1,5))
        rm = cv2.morphologyEx(gray, cv2.MORPH_OPEN, km)
        return cv2.inpaint(img, (255-rm).astype(np.uint8),3,cv2.INPAINT_TELEA)
    def _desnow(self, img):
        blur = cv2.GaussianBlur(img,(5,5),0)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        _, sm = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
        return cv2.inpaint(img,sm,5,cv2.INPAINT_NS)

# ======================== LiDAR处理 ========================
class LiDARProc:
    def __init__(self, cfg):
        self.cfg = cfg; self.q = queue.Queue(maxsize=2); self.data, self.trans = None, None
    def cb(self, pc):
        try:
            self.data = np.frombuffer(pc.raw_data, dtype=np.float32).reshape(-1,4)[:,:3]
            self.trans = pc.transform
            if self.q.full(): self.q.get_nowait()
            self.q.put((self.data.copy(), self.trans))
        except: pass
    def detect(self):
        if self.data is None or len(self.data)<50: return []
        gm = self.data[:,2] < -1.0; ng = self.data[~gm]
        if len(ng)<50: return []
        cls = DBSCAN(eps=0.8, min_samples=30).fit(ng[:,:2])
        boxes = []
        for l in set(cls.labels_):
            if l == -1: continue
            cp = ng[cls.labels_==l]
            if len(cp)<30: continue
            mn = cp.min(axis=0); mx = cp.max(axis=0)
            boxes.append({'3d_bbox':[mn[0],mn[1],mn[2],mx[0],mx[1],mx[2]],'center':[(mn[0]+mx[0])/2,(mn[1]+mx[1])/2,(mn[2]+mx[2])/2],'size':[mx[0]-mn[0],mx[1]-mn[1],mx[2]-mn[2]],'num_points':len(cp)})
        return boxes
    def get_3d(self):
        if self.data is None: return None
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.data)
        z_min,z_max = self.data[:,2].min(),self.data[:,2].max()
        c = (self.data[:,2]-z_min)/(z_max-z_min+1e-6)
        cm = np.zeros((len(c),3)); cm[:,0]=1-c; cm[:,2]=c
        pcd.colors = o3d.utility.Vector3dVector(cm)
        return pcd

# ======================== 数据记录 ========================
class Recorder:
    def __init__(self, cfg):
        self.cfg = cfg; self.dir = os.path.join(cfg.record_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.fr = 0; self.fs = {}
        if cfg.record_data:
            os.makedirs(self.dir, exist_ok=True)
            os.makedirs(os.path.join(self.dir,"screenshots"), exist_ok=True)
            self._init()
    def _init(self):
        tp = os.path.join(self.dir, "track_results.csv")
        self.fs['tracks'] = open(tp, 'w', newline='', encoding='utf-8')
        self.tw = csv.writer(self.fs['tracks'])
        self.tw.writerow(['timestamp','frame_id','track_id','x1','y1','x2','y2','cls_id','cls_name','behavior','speed','confidence'])
        pp = os.path.join(self.dir, "performance.csv")
        self.fs['performance'] = open(pp, 'w', newline='', encoding='utf-8')
        self.pw = csv.writer(self.fs['performance'])
        self.pw.writerow(['timestamp','frame_id','fps','cpu_usage','memory_usage','gpu_usage','detection_count','track_count'])
        cp = os.path.join(self.dir, "config.yaml")
        with open(cp, 'w', encoding='utf-8') as f: yaml.dump(self.cfg.__dict__, f, indent=2)
    def record(self, tracks, dets, fps):
        if not self.cfg.record_data: return
        if self.fr % (self.cfg.display_fps//self.cfg.record_fps) !=0:
            self.fr +=1; return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        if tracks and len(tracks)>0:
            for t in tracks:
                try:
                    b = "stopped" if t.is_stopped else "overtaking" if t.is_overtaking else "lane_changing" if t.is_lane_changing else "braking" if t.is_braking else "dangerous" if t.is_dangerous else "normal"
                    s = t._calc_speed() if hasattr(t,'_calc_speed') else 0.0
                    self.tw.writerow([ts,self.fr,t.track_id,t.bbox[0],t.bbox[1],t.bbox[2],t.bbox[3],t.cls_id,VEHICLE_CLS.get(t.cls_id,"Unknown"),b,s,t.conf if hasattr(t,'conf') else 0.0])
                except: pass
        cpu = psutil.cpu_percent(); mem = psutil.virtual_memory().percent
        gpu = torch.cuda.utilization() if (torch.cuda.is_available() and hasattr(torch.cuda,'utilization')) else 0.0
        self.pw.writerow([ts,self.fr,fps,cpu,mem,gpu,len(dets) if dets is not None else 0,len(tracks) if tracks is not None else 0])
        for f in self.fs.values(): f.flush()
        self.fr +=1
    def save_ss(self, img, w):
        if not self.cfg.save_screenshots or not valid_img(img): return
        p = os.path.join(self.dir,"screenshots",f"screenshot_{w}_{self.fr:06d}.png")
        cv2.imwrite(p, img)
    def close(self):
        if self.cfg.record_data:
            for f in self.fs.values():
                try: f.close()
                except: pass

# ======================== 卡尔曼滤波 ========================
class KF:
    def __init__(self, dt=0.05, ms=50.0):
        self.dt = dt; self.ms = ms; self.x = np.zeros(8, dtype=np.float32)
        self.F = np.array([[1,0,0,0,dt,0,0,0],[0,1,0,0,0,dt,0,0],[0,0,1,0,0,0,dt,0],[0,0,0,1,0,0,0,dt],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]], dtype=np.float32)
        self.H = np.eye(4,8,dtype=np.float32)
        self.Q = np.diag([1,1,1,1,5,5,5,5]).astype(np.float32)
        self.R = np.diag([5,5,5,5]).astype(np.float32)
        self.P = np.eye(8,dtype=np.float32)*50
    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4]
    def update(self, z):
        z = z.astype(np.float32)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        S_inv = np.linalg.pinv(S) if np.linalg.det(S)<1e-6 else np.linalg.inv(S)
        K = self.P @ self.H.T @ S_inv
        self.x = self.x + K @ y
        self.P = (np.eye(8)-K@self.H) @ self.P
        return self.x[:4]
    def update_noise(self, s):
        sf = min(1.0, s/self.ms)
        self.Q = np.diag([1+sf*4]*4 + [5+sf*20]*4).astype(np.float32)

# ======================== IOU计算 ========================
@njit
def iou(box1, box2):
    ix1 = max(box1[0], box2[0]); iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2]); iy2 = min(box1[3], box2[3])
    ia = max(0, ix2-ix1)*max(0, iy2-iy1)
    a1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    a2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    ua = a1+a2-ia
    return ia/ua if ua>0 else 0

# ======================== 跟踪目标 ========================
class Track:
    def __init__(self, tid, bbox, img_shape, kf_cfg, cfg):
        self.track_id = tid; self.kf = KF(dt=kf_cfg["dt"], ms=kf_cfg["max_speed"])
        self.img_shape = img_shape; self.cfg = cfg; self.bbox = clip_box(bbox.astype(np.float32), img_shape)
        self.kf.x[:4] = self.bbox; self.track_hist = []; self.speed_hist = []; self.accel_hist = []
        self.heading_hist = []; self.lat_dis = []; self.pred_traj = []
        self.is_stopped = self.is_overtaking = self.is_lane_changing = self.is_braking = self.is_accelerating = self.is_turning = self.is_dangerous = False
        self.stop_fr = self.overtake_fr = self.lane_change_fr = self.brake_fr = self.turn_fr = 0
        self.hits = 1; self.age = 0; self.tsu = 0; self.cls_id = None; self.conf = 0.0
        self._update_hist()
    def _update_hist(self):
        cx = (self.bbox[0]+self.bbox[2])/2; cy = (self.bbox[1]+self.bbox[3])/2
        self.track_hist.append((cx, cy))
        if len(self.track_hist)>1: self.lat_dis.append(abs(cx - self.track_hist[-2][0]))
        if len(self.track_hist)>self.cfg.track_history_len: self.track_hist.pop(0)
        if len(self.lat_dis)>10: self.lat_dis.pop(0)
    def _calc_speed(self):
        if len(self.track_hist)<2: return 0.0
        pc = self.track_hist[-2]; cc = self.track_hist[-1]
        s = np.linalg.norm(np.array(cc)-np.array(pc))/self.kf.dt
        self.speed_hist.append(s)
        if len(self.speed_hist)>1: self.accel_hist.append((s-self.speed_hist[-2])/self.kf.dt)
        if len(self.speed_hist)>5: self.speed_hist.pop(0)
        if len(self.accel_hist)>5: self.accel_hist.pop(0)
        return np.mean(self.speed_hist) if self.speed_hist else 0.0
    def _calc_heading(self):
        if len(self.track_hist)<3: return 0.0
        dx = self.track_hist[-1][0]-self.track_hist[-3][0]
        dy = self.track_hist[-1][1]-self.track_hist[-3][1]
        h = np.degrees(np.arctan2(dy, dx))
        self.heading_hist.append(h)
        if len(self.heading_hist)>5: self.heading_hist.pop(0)
        return h
    def _pred_traj(self):
        self.pred_traj = []
        if len(self.track_hist)<5: return
        tkf = KF(dt=self.kf.dt, ms=self.kf.ms)
        tkf.x = self.kf.x.copy(); tkf.P = self.kf.P.copy()
        for _ in range(self.cfg.predict_frames):
            pb = tkf.predict()
            self.pred_traj.append(((pb[0]+pb[2])/2, (pb[1]+pb[3])/2))
    def _analyze_behavior(self, ego_center):
        s = self._calc_speed(); h = self._calc_heading()
        if s < self.cfg.stop_speed_thresh:
            self.stop_fr +=1; self.is_stopped = self.stop_fr >= self.cfg.stop_frames_thresh
        else: self.stop_fr = 0; self.is_stopped = False
        if ego_center and len(self.track_hist)>=2:
            d = np.linalg.norm(np.array(self.track_hist[-1])-np.array(ego_center))
            if d < self.cfg.overtake_dist_thresh:
                es = getattr(self, 'ego_speed', 0.0)
                if s > es*self.cfg.overtake_speed_ratio:
                    self.overtake_fr +=1; self.is_overtaking = self.overtake_fr >=3
                else: self.overtake_fr =0; self.is_overtaking = False
            else: self.overtake_fr =0; self.is_overtaking = False
        if len(self.lat_dis)>=5:
            al = np.mean(self.lat_dis[-5:])
            if al>self.cfg.lane_change_thresh:
                self.lane_change_fr +=1; self.is_lane_changing = self.lane_change_fr >=3
            else: self.lane_change_fr =0; self.is_lane_changing = False
        if len(self.accel_hist)>=3:
            aa = np.mean(self.accel_hist[-3:])
            if aa < -self.cfg.brake_accel_thresh:
                self.brake_fr +=1; self.is_braking = self.brake_fr >=2; self.is_accelerating = False
            elif aa > self.cfg.brake_accel_thresh:
                self.is_accelerating = True; self.is_braking = False; self.brake_fr =0
            else: self.is_braking = self.is_accelerating = False; self.brake_fr =0
        if len(self.heading_hist)>=3:
            hd = np.abs(self.heading_hist[-1]-self.heading_hist[-3])
            if hd>self.cfg.turn_angle_thresh:
                self.turn_fr +=1; self.is_turning = self.turn_fr >=2
            else: self.turn_fr =0; self.is_turning = False
        if ego_center:
            d = np.linalg.norm(np.array(self.track_hist[-1])-np.array(ego_center))
            self.is_dangerous = d < self.cfg.danger_dist_thresh
        self._pred_traj()
    def predict(self):
        if len(self.track_hist)>=2:
            pc = np.array([(self.kf.x[0]+self.kf.x[2])/2, (self.kf.x[1]+self.kf.x[3])/2])
            cc = np.array([(self.bbox[0]+self.bbox[2])/2, (self.bbox[1]+self.bbox[3])/2])
            ps = np.linalg.norm(cc-pc)/self.kf.dt
            mps = max(self.img_shape)/self.kf.dt
            s = min(1.0, ps/mps)*self.kf.ms
        else: s=0.0
        self.bbox = self.kf.predict()
        self.bbox = clip_box(self.bbox, self.img_shape)
        self._update_hist(); self.age +=1; self.tsu +=1; self.kf.update_noise(s)
        return self.bbox
    def update(self, bbox, cls_id, conf=0.0, ego_center=None):
        self.bbox = self.kf.update(clip_box(bbox, self.img_shape))
        self._update_hist(); self.hits +=1; self.tsu =0; self.cls_id = cls_id; self.conf = conf
        self._analyze_behavior(ego_center)

# ======================== SORT跟踪器 ========================
class SORT:
    def __init__(self, cfg):
        self.max_age = cfg.max_age; self.min_hits = cfg.min_hits; self.iou_th = cfg.iou_thres
        self.img_shape = (cfg.img_height, cfg.img_width); self.kf_cfg = {"dt":cfg.kf_dt, "max_speed":cfg.max_speed}
        self.cfg = cfg; self.tracks = []; self.next_id =1; self.ego_center = None; self.ego_speed =0.0
    def update(self, dets, ego_center=None, lidar_dets=None):
        self.ego_center = ego_center
        if dets is None or len(dets)==0:
            if lidar_dets and len(lidar_dets)>0 and self.cfg.fuse_lidar_vision:
                dets = self._lidar2d(lidar_dets)
            self.tracks = [t for t in self.tracks if t.tsu <= self.max_age]
            return np.array([]), np.array([]), np.array([])
        vd = []
        for d in dets:
            if len(d)>=6:
                x1,y1,x2,y2,conf,cls_id = d[:6]
                if conf>0 and x2>x1 and y2>y1: vd.append([x1,y1,x2,y2,conf,int(cls_id)])
        vd = np.array(vd, dtype=np.float32)
        if len(vd)==0:
            self.tracks = [t for t in self.tracks if t.tsu <= self.max_age]
            return np.array([]), np.array([]), np.array([])
        for t in self.tracks: t.predict()
        if len(self.tracks)==0:
            for d in vd:
                self.tracks.append(Track(self.next_id, d[:4], self.img_shape, self.kf_cfg, self.cfg))
                self.next_id +=1
            return np.array([]), np.array([]), np.array([])
        try:
            iou_mat = np.array([[iou(t.bbox, d[:4]) for t in self.tracks] for d in vd])
            cost_mat = 1 - iou_mat
            t_idx, d_idx = linear_sum_assignment(cost_mat)
        except: t_idx, d_idx = [], []
        matches, ud, ut = [], set(), set()
        for ti, di in zip(t_idx, d_idx):
            if ti<len(self.tracks) and di<len(vd) and iou(self.tracks[ti].bbox, vd[di][:4])>self.iou_th:
                matches.append((ti,di)); ud.add(di); ut.add(ti)
        for ti, di in matches:
            self.tracks[ti].update(vd[di][:4], int(vd[di][5]), vd[di][4], self.ego_center)
            self.tracks[ti].ego_speed = self.ego_speed
        for di in set(range(len(vd)))-ud:
            self.tracks.append(Track(self.next_id, vd[di][:4], self.img_shape, self.kf_cfg, self.cfg))
            self.next_id +=1
        self.tracks = [t for t in self.tracks if t.tsu <= self.max_age]
        vt = [t for t in self.tracks if t.hits >= self.min_hits]
        if not vt: return np.array([]), np.array([]), np.array([])
        boxes = np.array([t.bbox.astype(int) for t in vt])
        ids = np.array([t.track_id for t in vt])
        cls = np.array([t.cls_id if t.cls_id is not None else -1 for t in vt])
        return boxes, ids, cls
    def _lidar2d(self, lidar_dets):
        dets = []
        for d in lidar_dets:
            c = d['center']; s = d['size']
            x1 = c[0]*10 + self.img_shape[1]/2; y1 = c[1]*10 + self.img_shape[0]/2
            x2 = x1 + s[0]*5; y2 = y1 + s[1]*5
            dets.append([x1,y1,x2,y2,0.8,2])
        return np.array(dets)

# ======================== 检测线程 ========================
class DetThread(threading.Thread):
    def __init__(self, det, cfg, enh, in_q, out_q, dev="cpu"):
        super().__init__(daemon=True)
        self.det = det; self.cfg = cfg; self.enh = enh; self.in_q = in_q; self.out_q = out_q
        self.running = True; self.dev = dev
    def run(self):
        while self.running:
            try:
                img = self.in_q.get(timeout=1.0)
                if not valid_img(img):
                    self.out_q.put((None, np.array([])))
                    continue
                img_enh = self.enh.enhance(img)
                h,w = img.shape[:2]; r = min(self.cfg.yolo_imgsz_max/w, self.cfg.yolo_imgsz_max/h)
                ws, hs = make_div(int(w*r)), make_div(int(h*r))
                res = self.det.predict(img_enh, conf=self.cfg.conf_thres, verbose=False, device=self.dev, agnostic_nms=True, imgsz=(hs,ws), iou=self.cfg.yolo_iou)
                dets = []
                for r in res:
                    if hasattr(r, 'boxes') and r.boxes is not None and len(r.boxes)>0:
                        for b in r.boxes:
                            if b.cls is not None and b.conf is not None and b.xyxy is not None:
                                cid = int(b.cls[0])
                                if cid in {2,5,7}:
                                    xyxy = b.xyxy[0].cpu().numpy()
                                    conf = float(b.conf[0])
                                    if xyxy[2]>xyxy[0] and xyxy[3]>xyxy[1] and conf>0:
                                        dets.append([*xyxy, conf, cid])
                self.out_q.put((img, np.array(dets, dtype=np.float32)))
            except queue.Empty: continue
            except: self.out_q.put((None, np.array([])))
    def stop(self): self.running = False

# ======================== 可视化工具（核心：优化检测框绘制） ========================
class FrameBuf:
    def __init__(self, sz=(480,640,3)):
        self.df = np.zeros(sz, dtype=np.uint8)
        cv2.putText(self.df, "Initializing...", (100,240), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        self.cf = self.df.copy(); self.lock = threading.Lock()
    def update(self, f):
        if valid_img(f):
            with self.lock: self.cf = f.copy()
    def get(self): return self.cf.copy()

# 新增：绘制缓存类，提升性能
class DrawCache:
    """绘制缓存类，减少重复计算"""
    def __init__(self, cfg):
        self.cfg = cfg
        self.label_cache = {}  # 缓存标签尺寸
        self.color_cache = {}  # 缓存跟踪ID颜色

    def get_label_size(self, text):
        """缓存标签文字尺寸"""
        if text not in self.label_cache:
            size = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX,
                self.cfg.det_label_font_scale, 1
            )[0]
            self.label_cache[text] = size
        return self.label_cache[text]

    def get_track_color(self, track_id):
        """缓存跟踪ID颜色"""
        if track_id not in self.color_cache:
            self.color_cache[track_id] = ((track_id*59)%256, (track_id*127)%256, (track_id*199)%256)
        return self.color_cache[track_id]

class FPS:
    def __init__(self, ws=15):
        self.ws = ws; self.times = []; self.fps =0.0
    def update(self):
        self.times.append(time.time())
        if len(self.times)>self.ws: self.times.pop(0)
        if len(self.times)>=2: self.fps = (len(self.times)-1)/(self.times[-1]-self.times[0])
        return self.fps

# 核心修改：自定义检测框绘制（匹配截图样式）
def draw(img, boxes, ids, cls_ids, tracks, fps=0.0, det_cnt=0, cfg=None, w="clear", perf=None):
    if not valid_img(img): 
        return np.zeros((480,640,3), dtype=np.uint8)
    cfg = cfg or Config()
    di = img.copy()
    draw_cache = DrawCache(cfg)  # 初始化绘制缓存
    
    # 1. 绘制顶部信息栏（性能/天气/计数）
    ov = di.copy()
    cv2.rectangle(ov, (10,10), (800,80), (0,0,0), -1)
    cv2.addWeighted(ov,0.7,di,0.3,0,di)
    
    sc = sum(1 for t in tracks if t.is_stopped) if tracks else 0
    oc = sum(1 for t in tracks if t.is_overtaking) if tracks else 0
    lc = sum(1 for t in tracks if t.is_lane_changing) if tracks else 0
    bc = sum(1 for t in tracks if t.is_braking) if tracks else 0
    dc = sum(1 for t in tracks if t.is_dangerous) if tracks else 0
    
    lines = [
        f"FPS:{fps:.1f} | Weather:{w} | Tracks:{len(boxes)} | Dets:{det_cnt}",
        f"Stop:{sc} | Overtake:{oc} | LaneChange:{lc} | Brake:{bc} | Danger:{dc}"
    ]
    if perf: 
        lines.append(f"CPU:{perf['cpu']:.1f}% | MEM:{perf['mem']:.1f}% | GPU:{perf['gpu']:.1f}%")
    
    for i, l in enumerate(lines):
        cv2.putText(di, l, (15,30+i*20), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2,cv2.LINE_AA)

    # 2. 绘制车辆检测框（核心：匹配截图样式）
    if boxes is not None and ids is not None and cls_ids is not None and tracks is not None:
        ml = min(len(boxes), len(ids), len(cls_ids), len(tracks))
        for i in range(ml):
            try:
                b = boxes[i]
                tid = ids[i]
                cid = cls_ids[i]
                t = tracks[i]
                
                # 过滤无效框
                if b is None or len(b)!=4 or b[0]>=b[2] or b[1]>=b[3]: 
                    continue
                
                x1,y1,x2,y2 = map(int, b)
                cls_name = VEHICLE_CLS.get(cid, "Unknown")
                conf = t.conf if hasattr(t, 'conf') else 0.0
                
                # ========== 核心：自定义检测框（匹配截图） ==========
                # 1. 绘制蓝色检测框（无透明度，实线）
                cv2.rectangle(di, (x1, y1), (x2, y2), cfg.det_box_color, cfg.det_box_thickness, cv2.LINE_AA)
                
                # 2. 绘制标签（类别+置信度，黑底白字）
                label = f"{cls_name} {conf:.2f}"
                # 计算标签尺寸（缓存复用）
                label_w, label_h = draw_cache.get_label_size(label)
                # 绘制标签背景（紧贴框体顶部）
                label_x1 = x1
                label_y1 = y1 - label_h - 2*cfg.det_label_padding
                label_x2 = x1 + label_w + 2*cfg.det_label_padding
                label_y2 = y1
                cv2.rectangle(
                    di, (label_x1, label_y1), (label_x2, label_y2),
                    cfg.det_label_bg_color, -1  # 实心背景
                )
                # 绘制标签文字
                cv2.putText(
                    di, label, 
                    (x1 + cfg.det_label_padding, y1 - cfg.det_label_padding),
                    cv2.FONT_HERSHEY_SIMPLEX, cfg.det_label_font_scale,
                    cfg.det_label_text_color, 1, cv2.LINE_AA
                )

                # 3. 绘制跟踪ID（可选，绿色背景）
                id_label = f"ID:{tid}"
                id_w, id_h = cv2.getTextSize(id_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                cv2.rectangle(
                    di, (x1, label_y1 - id_h - 2*cfg.det_label_padding),
                    (x1 + id_w + 2*cfg.det_label_padding, label_y1),
                    (0, 255, 0), -1
                )
                cv2.putText(
                    di, id_label,
                    (x1 + cfg.det_label_padding, label_y1 - cfg.det_label_padding),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA
                )

                # 4. 行为标签（简化显示，红色背景）
                behavior_tags = []
                if t.is_stopped: behavior_tags.append("STOP")
                if t.is_dangerous: behavior_tags.append("DANGER")
                if behavior_tags:
                    behavior_label = " | ".join(behavior_tags)
                    bh_w, bh_h = cv2.getTextSize(behavior_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                    cv2.rectangle(
                        di, (x2 - bh_w - 2*cfg.det_label_padding, y1 - bh_h - 2*cfg.det_label_padding),
                        (x2, y1), (0,0,255), -1
                    )
                    cv2.putText(
                        di, behavior_label,
                        (x2 - bh_w - cfg.det_label_padding, y1 - cfg.det_label_padding),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA
                    )

                # 5. 简化轨迹绘制（只绘制最近5帧，提升性能）
                if t and len(t.track_hist)>=2:
                    hist = t.track_hist[-5:]
                    for j in range(1, len(hist)):
                        p1 = (int(hist[j-1][0]), int(hist[j-1][1]))
                        p2 = (int(hist[j][0]), int(hist[j][1]))
                        cv2.line(di, p1, p2, (0,255,0), 1, cv2.LINE_AA)

            except Exception as e:
                logger.warning(f"绘制检测框失败: {e}")
                continue

    return di

# ======================== 工具函数 ========================
def clear_actors(world, exclude=None):
    ei = set(exclude) if exclude else set()
    actors = world.get_actors()
    actor_id_map = {a.id: a for a in actors if a.is_alive}
    for at in ['vehicle.', 'sensor.']:
        al = [a for a_id, a in actor_id_map.items() if a.type_id.startswith(at) and a_id not in ei]
        for i in range(0, len(al),10):
            for a in al[i:i+10]:
                try:
                    if a.is_alive and a.id in actor_id_map:
                        a.destroy()
                        del actor_id_map[a.id]
                except Exception as e:
                    logger.warning(f"跳过销毁Actor {a.id}：{str(e)[:30]}")

def cam_cb(img, q):
    try:
        ia = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width,4))
        ir = cv2.GaussianBlur(ia[:,:,:3], (3,3),0)
        if q.full(): q.get_nowait()
        q.put(ir)
    except: pass

def spawn_npcs(world, num, sp):
    bps = [bp for bp in world.get_blueprint_library().filter('vehicle') if int(bp.get_attribute('number_of_wheels'))==4 and not bp.id.endswith(('firetruck','ambulance','police'))]
    if not bps: return 0
    cnt, used, max_att = 0, set(), num*3
    for _ in range(max_att):
        if cnt>=num or len(used)>=len(sp): break
        p = random.choice(sp)
        k = (round(p.location.x,2), round(p.location.y,2), round(p.location.z,2))
        if k not in used:
            used.add(k)
            n = world.try_spawn_actor(random.choice(bps), p)
            if n:
                try: n.set_autopilot(True, tm_port=8000)
                except: n.set_autopilot(True)
                cnt +=1
    return cnt

# 安全生成自车函数
def safe_spawn_ego(world, spawn_points):
    """安全生成自车，避免碰撞"""
    ego_bp = random.choice(world.get_blueprint_library().filter('vehicle.tesla.model3'))
    ego_bp.set_attribute('color', '255,0,0')
    # 遍历生成点，直到找到无碰撞的位置
    for spawn_point in spawn_points:
        ego = world.try_spawn_actor(ego_bp, spawn_point)
        if ego is not None:
            print(f"✅ 自车生成成功，位置：{spawn_point.location}")
            return ego
    # 若所有生成点都碰撞，随机偏移位置重试
    print("⚠️ 所有默认生成点有碰撞，尝试偏移位置...")
    for spawn_point in spawn_points:
        # 随机偏移x/y坐标（±2米）
        spawn_point.location.x += random.uniform(-2, 2)
        spawn_point.location.y += random.uniform(-2, 2)
        ego = world.try_spawn_actor(ego_bp, spawn_point)
        if ego is not None:
            print(f"✅ 自车生成成功（偏移位置）：{spawn_point.location}")
            return ego
    print("❌ 无法生成自车，所有位置都有碰撞")
    return None

# ======================== 主函数 ========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="配置文件路径")
    parser.add_argument("--host", help="CARLA主机")
    parser.add_argument("--port", type=int, help="CARLA端口")
    parser.add_argument("--conf-thres", type=float, help="检测置信度")
    parser.add_argument("--weather", help="初始天气")
    args = parser.parse_args()
    
    cfg = Config.from_yaml(args.config)
    if args.host: cfg.host = args.host
    if args.port: cfg.port = args.port
    if args.conf_thres: cfg.conf_thres = args.conf_thres
    if args.weather and args.weather in WEATHER: cfg.default_weather = args.weather

    # 初始化所有可能用到的变量
    ego = None
    cam = None
    lidar = None
    lidar_proc = None
    det_thread = None
    vis = None
    recorder = Recorder(cfg)

    try:
        client = carla.Client(cfg.host, cfg.port)
        client.set_timeout(20.0)
        world = client.get_world()
        try:
            tm = client.get_trafficmanager(8000)
            tm.set_global_distance_to_leading_vehicle(2.0)
            tm.set_respawn_dormant_vehicles(True)
            tm.set_hybrid_physics_mode(True)
            tm.set_hybrid_physics_radius(50.0)
            tm.global_percentage_speed_difference(0)
        except: pass
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 10
        world.apply_settings(settings)

        # 设置初始天气
        world.set_weather(WEATHER[cfg.default_weather])
        we = WeatherEnhancer(cfg); we.set_weather(cfg.default_weather); cw = cfg.default_weather

        # 获取生成点并安全生成自车
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            print("❌ 无可用生成点")
            return
        
        ego = safe_spawn_ego(world, spawn_points)
        if ego is None:
            return
        ego.set_autopilot(True, tm_port=8000)

        # 生成NPC
        npc_count = spawn_npcs(world, cfg.num_npcs, spawn_points)
        print(f"✅ 生成NPC车辆：{npc_count} 辆")

        # 初始化相机
        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(cfg.img_width))
        cam_bp.set_attribute('image_size_y', str(cfg.img_height))
        cam_bp.set_attribute('fov', '90')
        cam_bp.set_attribute('sensor_tick', '0.05')
        cam_t = carla.Transform(carla.Location(x=2.0, z=1.8))
        cam = world.spawn_actor(cam_bp, cam_t, attach_to=ego)
        cam_q = queue.Queue(maxsize=1)
        cam.listen(lambda img: cam_cb(img, cam_q))
        print(f"✅ 相机传感器启动成功 (ID: {cam.id})")

        # 初始化LiDAR
        if cfg.use_lidar:
            lidar_proc = LiDARProc(cfg)
            lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
            lidar_bp.set_attribute('channels', str(cfg.lidar_channels))
            lidar_bp.set_attribute('range', str(cfg.lidar_range))
            lidar_bp.set_attribute('points_per_second', str(cfg.lidar_points_per_second))
            lidar_bp.set_attribute('rotation_frequency', '20')
            lidar_bp.set_attribute('sensor_tick', '0.05')
            lidar_t = carla.Transform(carla.Location(x=0.0, z=2.5))
            lidar = world.spawn_actor(lidar_bp, lidar_t, attach_to=ego)
            lidar.listen(lidar_proc.cb)
            print(f"✅ LiDAR传感器启动成功 (ID: {lidar.id})")

        # 初始化YOLO
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✅ 使用设备: {dev}")
        model = YOLO(cfg.yolo_model)
        if cfg.yolo_quantize and dev=="cuda": 
            model = model.quantize()
            print("✅ YOLO模型已量化")

        # 启动检测线程
        in_q = queue.Queue(maxsize=2); out_q = queue.Queue(maxsize=2)
        det_thread = DetThread(model, cfg, we, in_q, out_q, dev)
        det_thread.start()
        print("✅ 推理线程已启动")

        # 初始化跟踪器和可视化
        tracker = SORT(cfg)
        fb = FrameBuf((cfg.img_height, cfg.img_width, 3))
        fps_cnt = FPS(cfg.fps_window_size)
        cv2.namedWindow("CARLA Object Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("CARLA Object Tracking", cfg.window_width, cfg.window_height)

        # 初始化3D可视化
        if cfg.use_3d_visualization and cfg.use_lidar and lidar_proc is not None:
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="LiDAR Point Cloud", width=cfg.pcd_view_size, height=cfg.pcd_view_size)
            print("✅ 3D点云可视化窗口已启动")
        else:
            vis = None

        # 主循环
        print("🚀 开始跟踪（按ESC退出，按W切换天气）")
        fr_cnt = 0
        last_display_time = time.time()
        while True:
            world.tick()
            
            # 帧率控制
            current_time = time.time()
            elapsed = current_time - last_display_time
            target_interval = 1.0 / cfg.display_fps
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)
            last_display_time = current_time
            
            # 获取相机图像
            try:
                img = cam_q.get(timeout=0.1)
                fb.update(img)
            except: 
                img = fb.get()
            
            # 提交推理任务
            if not in_q.full(): 
                in_q.put(img.copy())
            
            # 获取检测结果
            dets = np.array([])
            try: 
                _, dets = out_q.get_nowait()
            except: 
                pass
            
            # LiDAR检测
            lidar_dets = lidar_proc.detect() if (cfg.use_lidar and lidar_proc) else []
            
            # 更新跟踪器
            boxes, ids, cls_ids = tracker.update(dets, (cfg.img_width//2, cfg.img_height//2), lidar_dets)
            
            # 性能计算
            fps = fps_cnt.update()
            cpu = psutil.cpu_percent(); mem = psutil.virtual_memory().percent
            gpu = torch.cuda.utilization() if (torch.cuda.is_available() and hasattr(torch.cuda,'utilization')) else 0.0
            perf = {'cpu':cpu, 'mem':mem, 'gpu':gpu, 'fps':fps, 'avg_fps':fps}
            
            # 绘制可视化（使用新的检测框样式）
            di = draw(img, boxes, ids, cls_ids, tracker.tracks, fps=fps, det_cnt=len(dets), cfg=cfg, w=cw, perf=perf)
            cv2.imshow("CARLA Object Tracking", di)
            
            # 3D点云更新
            if cfg.use_3d_visualization and vis and lidar_proc:
                pcd = lidar_proc.get_3d()
                if pcd:
                    vis.clear_geometries()
                    vis.add_geometry(pcd)
                    vis.poll_events()
                    vis.update_renderer()
            
            # 数据记录
            recorder.record(tracker.tracks, dets, fps)
            if cfg.save_screenshots and fr_cnt%30==0:
                recorder.save_ss(di, cw)
            
            # 键盘事件处理
            key = cv2.waitKey(1) & 0xFF
            if key ==27: 
                print("🛑 用户按下ESC，退出程序")
                break
            elif key == ord('w') or key == ord('W'):
                wl = list(WEATHER.keys())
                cwi = wl.index(cw)
                cw = wl[(cwi+1)%len(wl)]
                world.set_weather(WEATHER[cw])
                we.set_weather(cw)
                print(f"🌤️ 已切换天气到: {cw} (可选：{wl})")
            fr_cnt +=1

    except KeyboardInterrupt: 
        print("🛑 用户中断程序")
    except Exception as e: 
        print(f"❌ 运行错误: {str(e)}")
    finally:
        # 资源清理
        print("🧹 开始清理资源...")
        
        # 停止检测线程
        if det_thread:
            det_thread.stop()
            det_thread.join(timeout=2.0)
        
        # 关闭3D可视化窗口
        if vis is not None:
            try:
                vis.destroy_window()
            except:
                pass
        
        # 关闭OpenCV窗口
        cv2.destroyAllWindows()
        
        # 关闭数据记录
        recorder.close()
        
        # 销毁LiDAR
        if lidar and lidar.is_alive:
            try: 
                lidar.stop()
                lidar.destroy()
                print("✅ LiDAR已销毁")
            except Exception as e:
                print(f"⚠️ 销毁LiDAR失败: {e}")
        
        # 销毁相机
        if cam and cam.is_alive:
            try: 
                cam.stop()
                cam.destroy()
                print("✅ 相机已销毁")
            except Exception as e:
                print(f"⚠️ 销毁相机失败: {e}")
        
        # 销毁自车
        if ego and ego.is_alive:
            try: 
                ego.destroy()
                print("✅ 自车已销毁")
            except Exception as e:
                print(f"⚠️ 销毁自车失败: {e}")
        
        # 清理所有NPC和传感器
        clear_actors(world)
        
        # 恢复CARLA设置
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print("✅ 所有资源清理完成")

if __name__ == "__main__":
    # 配置日志
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>", level="INFO")
    logger.add(f"track_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", rotation="100 MB", retention="7 days", format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}", level="DEBUG")
    
    # 启动主程序
    main()