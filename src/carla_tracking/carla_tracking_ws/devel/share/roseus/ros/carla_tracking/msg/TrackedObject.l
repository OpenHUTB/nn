;; Auto-generated. Do not edit!


(when (boundp 'carla_tracking::TrackedObject)
  (if (not (find-package "CARLA_TRACKING"))
    (make-package "CARLA_TRACKING"))
  (shadow 'TrackedObject (find-package "CARLA_TRACKING")))
(unless (find-package "CARLA_TRACKING::TRACKEDOBJECT")
  (make-package "CARLA_TRACKING::TRACKEDOBJECT"))

(in-package "ROS")
;;//! \htmlinclude TrackedObject.msg.html


(defclass carla_tracking::TrackedObject
  :super ros::object
  :slots (_x1 _y1 _x2 _y2 _track_id _class_id _confidence _distance _velocity ))

(defmethod carla_tracking::TrackedObject
  (:init
   (&key
    ((:x1 __x1) 0.0)
    ((:y1 __y1) 0.0)
    ((:x2 __x2) 0.0)
    ((:y2 __y2) 0.0)
    ((:track_id __track_id) 0)
    ((:class_id __class_id) 0)
    ((:confidence __confidence) 0.0)
    ((:distance __distance) 0.0)
    ((:velocity __velocity) 0.0)
    )
   (send-super :init)
   (setq _x1 (float __x1))
   (setq _y1 (float __y1))
   (setq _x2 (float __x2))
   (setq _y2 (float __y2))
   (setq _track_id (round __track_id))
   (setq _class_id (round __class_id))
   (setq _confidence (float __confidence))
   (setq _distance (float __distance))
   (setq _velocity (float __velocity))
   self)
  (:x1
   (&optional __x1)
   (if __x1 (setq _x1 __x1)) _x1)
  (:y1
   (&optional __y1)
   (if __y1 (setq _y1 __y1)) _y1)
  (:x2
   (&optional __x2)
   (if __x2 (setq _x2 __x2)) _x2)
  (:y2
   (&optional __y2)
   (if __y2 (setq _y2 __y2)) _y2)
  (:track_id
   (&optional __track_id)
   (if __track_id (setq _track_id __track_id)) _track_id)
  (:class_id
   (&optional __class_id)
   (if __class_id (setq _class_id __class_id)) _class_id)
  (:confidence
   (&optional __confidence)
   (if __confidence (setq _confidence __confidence)) _confidence)
  (:distance
   (&optional __distance)
   (if __distance (setq _distance __distance)) _distance)
  (:velocity
   (&optional __velocity)
   (if __velocity (setq _velocity __velocity)) _velocity)
  (:serialization-length
   ()
   (+
    ;; float32 _x1
    4
    ;; float32 _y1
    4
    ;; float32 _x2
    4
    ;; float32 _y2
    4
    ;; int32 _track_id
    4
    ;; int32 _class_id
    4
    ;; float32 _confidence
    4
    ;; float32 _distance
    4
    ;; float32 _velocity
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32 _x1
       (sys::poke _x1 (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _y1
       (sys::poke _y1 (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _x2
       (sys::poke _x2 (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _y2
       (sys::poke _y2 (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; int32 _track_id
       (write-long _track_id s)
     ;; int32 _class_id
       (write-long _class_id s)
     ;; float32 _confidence
       (sys::poke _confidence (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _distance
       (sys::poke _distance (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _velocity
       (sys::poke _velocity (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32 _x1
     (setq _x1 (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _y1
     (setq _y1 (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _x2
     (setq _x2 (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _y2
     (setq _y2 (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; int32 _track_id
     (setq _track_id (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; int32 _class_id
     (setq _class_id (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; float32 _confidence
     (setq _confidence (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _distance
     (setq _distance (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _velocity
     (setq _velocity (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get carla_tracking::TrackedObject :md5sum-) "6680634a5eec139a85714bade9173e43")
(setf (get carla_tracking::TrackedObject :datatype-) "carla_tracking/TrackedObject")
(setf (get carla_tracking::TrackedObject :definition-)
      "# 跟踪目标自定义消息
float32 x1          # 边界框左上角x
float32 y1          # 边界框左上角y
float32 x2          # 边界框右下角x
float32 y2          # 边界框右下角y
int32 track_id      # 跟踪ID
int32 class_id      # 类别ID
float32 confidence  # 置信度
float32 distance    # 距离（米）
float32 velocity    # 速度

")



(provide :carla_tracking/TrackedObject "6680634a5eec139a85714bade9173e43")


