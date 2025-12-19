;; Auto-generated. Do not edit!


(when (boundp 'carla_autonomous::Stop)
  (if (not (find-package "CARLA_AUTONOMOUS"))
    (make-package "CARLA_AUTONOMOUS"))
  (shadow 'Stop (find-package "CARLA_AUTONOMOUS")))
(unless (find-package "CARLA_AUTONOMOUS::STOP")
  (make-package "CARLA_AUTONOMOUS::STOP"))
(unless (find-package "CARLA_AUTONOMOUS::STOPREQUEST")
  (make-package "CARLA_AUTONOMOUS::STOPREQUEST"))
(unless (find-package "CARLA_AUTONOMOUS::STOPRESPONSE")
  (make-package "CARLA_AUTONOMOUS::STOPRESPONSE"))

(in-package "ROS")





(defclass carla_autonomous::StopRequest
  :super ros::object
  :slots ())

(defmethod carla_autonomous::StopRequest
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

(defclass carla_autonomous::StopResponse
  :super ros::object
  :slots (_success _message ))

(defmethod carla_autonomous::StopResponse
  (:init
   (&key
    ((:success __success) nil)
    ((:message __message) "")
    )
   (send-super :init)
   (setq _success __success)
   (setq _message (string __message))
   self)
  (:success
   (&optional (__success :null))
   (if (not (eq __success :null)) (setq _success __success)) _success)
  (:message
   (&optional __message)
   (if __message (setq _message __message)) _message)
  (:serialization-length
   ()
   (+
    ;; bool _success
    1
    ;; string _message
    4 (length _message)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; bool _success
       (if _success (write-byte -1 s) (write-byte 0 s))
     ;; string _message
       (write-long (length _message) s) (princ _message s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; bool _success
     (setq _success (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; string _message
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _message (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass carla_autonomous::Stop
  :super ros::object
  :slots ())

(setf (get carla_autonomous::Stop :md5sum-) "937c9679a518e3a18d831e57125ea522")
(setf (get carla_autonomous::Stop :datatype-) "carla_autonomous/Stop")
(setf (get carla_autonomous::Stop :request) carla_autonomous::StopRequest)
(setf (get carla_autonomous::Stop :response) carla_autonomous::StopResponse)

(defmethod carla_autonomous::StopRequest
  (:response () (instance carla_autonomous::StopResponse :init)))

(setf (get carla_autonomous::StopRequest :md5sum-) "937c9679a518e3a18d831e57125ea522")
(setf (get carla_autonomous::StopRequest :datatype-) "carla_autonomous/StopRequest")
(setf (get carla_autonomous::StopRequest :definition-)
      "---
bool success
string message
")

(setf (get carla_autonomous::StopResponse :md5sum-) "937c9679a518e3a18d831e57125ea522")
(setf (get carla_autonomous::StopResponse :datatype-) "carla_autonomous/StopResponse")
(setf (get carla_autonomous::StopResponse :definition-)
      "---
bool success
string message
")



(provide :carla_autonomous/Stop "937c9679a518e3a18d831e57125ea522")


