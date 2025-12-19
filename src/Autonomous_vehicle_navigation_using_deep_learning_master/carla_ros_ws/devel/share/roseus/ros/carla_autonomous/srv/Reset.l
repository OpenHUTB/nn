;; Auto-generated. Do not edit!


(when (boundp 'carla_autonomous::Reset)
  (if (not (find-package "CARLA_AUTONOMOUS"))
    (make-package "CARLA_AUTONOMOUS"))
  (shadow 'Reset (find-package "CARLA_AUTONOMOUS")))
(unless (find-package "CARLA_AUTONOMOUS::RESET")
  (make-package "CARLA_AUTONOMOUS::RESET"))
(unless (find-package "CARLA_AUTONOMOUS::RESETREQUEST")
  (make-package "CARLA_AUTONOMOUS::RESETREQUEST"))
(unless (find-package "CARLA_AUTONOMOUS::RESETRESPONSE")
  (make-package "CARLA_AUTONOMOUS::RESETRESPONSE"))

(in-package "ROS")



(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass carla_autonomous::ResetRequest
  :super ros::object
  :slots ())

(defmethod carla_autonomous::ResetRequest
  (:init
   (&key
    )
   (send-super :init)
   self)
  (:serialization-length
   ()
   (+
    0
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;;
   self)
  )

(defclass carla_autonomous::ResetResponse
  :super ros::object
  :slots (_state ))

(defmethod carla_autonomous::ResetResponse
  (:init
   (&key
    ((:state __state) (instance std_msgs::Float32MultiArray :init))
    )
   (send-super :init)
   (setq _state __state)
   self)
  (:state
   (&rest __state)
   (if (keywordp (car __state))
       (send* _state __state)
     (progn
       (if __state (setq _state (car __state)))
       _state)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Float32MultiArray _state
    (send _state :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Float32MultiArray _state
       (send _state :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Float32MultiArray _state
     (send _state :deserialize buf ptr-) (incf ptr- (send _state :serialization-length))
   ;;
   self)
  )

(defclass carla_autonomous::Reset
  :super ros::object
  :slots ())

(setf (get carla_autonomous::Reset :md5sum-) "203da5fa4570387c68513683b6bcbd59")
(setf (get carla_autonomous::Reset :datatype-) "carla_autonomous/Reset")
(setf (get carla_autonomous::Reset :request) carla_autonomous::ResetRequest)
(setf (get carla_autonomous::Reset :response) carla_autonomous::ResetResponse)

(defmethod carla_autonomous::ResetRequest
  (:response () (instance carla_autonomous::ResetResponse :init)))

(setf (get carla_autonomous::ResetRequest :md5sum-) "203da5fa4570387c68513683b6bcbd59")
(setf (get carla_autonomous::ResetRequest :datatype-) "carla_autonomous/ResetRequest")
(setf (get carla_autonomous::ResetRequest :definition-)
      "---
std_msgs/Float32MultiArray state

================================================================================
MSG: std_msgs/Float32MultiArray
# Please look at the MultiArrayLayout message definition for
# documentation on all multiarrays.

MultiArrayLayout  layout        # specification of data layout
float32[]         data          # array of data


================================================================================
MSG: std_msgs/MultiArrayLayout
# The multiarray declares a generic multi-dimensional array of a
# particular data type.  Dimensions are ordered from outer most
# to inner most.

MultiArrayDimension[] dim # Array of dimension properties
uint32 data_offset        # padding elements at front of data

# Accessors should ALWAYS be written in terms of dimension stride
# and specified outer-most dimension first.
# 
# multiarray(i,j,k) = data[data_offset + dim_stride[1]*i + dim_stride[2]*j + k]
#
# A standard, 3-channel 640x480 image with interleaved color channels
# would be specified as:
#
# dim[0].label  = \"height\"
# dim[0].size   = 480
# dim[0].stride = 3*640*480 = 921600  (note dim[0] stride is just size of image)
# dim[1].label  = \"width\"
# dim[1].size   = 640
# dim[1].stride = 3*640 = 1920
# dim[2].label  = \"channel\"
# dim[2].size   = 3
# dim[2].stride = 3
#
# multiarray(i,j,k) refers to the ith row, jth column, and kth channel.

================================================================================
MSG: std_msgs/MultiArrayDimension
string label   # label of given dimension
uint32 size    # size of given dimension (in type units)
")

(setf (get carla_autonomous::ResetResponse :md5sum-) "203da5fa4570387c68513683b6bcbd59")
(setf (get carla_autonomous::ResetResponse :datatype-) "carla_autonomous/ResetResponse")
(setf (get carla_autonomous::ResetResponse :definition-)
      "---
std_msgs/Float32MultiArray state

================================================================================
MSG: std_msgs/Float32MultiArray
# Please look at the MultiArrayLayout message definition for
# documentation on all multiarrays.

MultiArrayLayout  layout        # specification of data layout
float32[]         data          # array of data


================================================================================
MSG: std_msgs/MultiArrayLayout
# The multiarray declares a generic multi-dimensional array of a
# particular data type.  Dimensions are ordered from outer most
# to inner most.

MultiArrayDimension[] dim # Array of dimension properties
uint32 data_offset        # padding elements at front of data

# Accessors should ALWAYS be written in terms of dimension stride
# and specified outer-most dimension first.
# 
# multiarray(i,j,k) = data[data_offset + dim_stride[1]*i + dim_stride[2]*j + k]
#
# A standard, 3-channel 640x480 image with interleaved color channels
# would be specified as:
#
# dim[0].label  = \"height\"
# dim[0].size   = 480
# dim[0].stride = 3*640*480 = 921600  (note dim[0] stride is just size of image)
# dim[1].label  = \"width\"
# dim[1].size   = 640
# dim[1].stride = 3*640 = 1920
# dim[2].label  = \"channel\"
# dim[2].size   = 3
# dim[2].stride = 3
#
# multiarray(i,j,k) refers to the ith row, jth column, and kth channel.

================================================================================
MSG: std_msgs/MultiArrayDimension
string label   # label of given dimension
uint32 size    # size of given dimension (in type units)
")



(provide :carla_autonomous/Reset "203da5fa4570387c68513683b6bcbd59")


