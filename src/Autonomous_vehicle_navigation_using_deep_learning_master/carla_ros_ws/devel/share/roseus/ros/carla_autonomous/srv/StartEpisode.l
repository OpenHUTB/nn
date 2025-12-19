;; Auto-generated. Do not edit!


(when (boundp 'carla_autonomous::StartEpisode)
  (if (not (find-package "CARLA_AUTONOMOUS"))
    (make-package "CARLA_AUTONOMOUS"))
  (shadow 'StartEpisode (find-package "CARLA_AUTONOMOUS")))
(unless (find-package "CARLA_AUTONOMOUS::STARTEPISODE")
  (make-package "CARLA_AUTONOMOUS::STARTEPISODE"))
(unless (find-package "CARLA_AUTONOMOUS::STARTEPISODEREQUEST")
  (make-package "CARLA_AUTONOMOUS::STARTEPISODEREQUEST"))
(unless (find-package "CARLA_AUTONOMOUS::STARTEPISODERESPONSE")
  (make-package "CARLA_AUTONOMOUS::STARTEPISODERESPONSE"))

(in-package "ROS")





(defclass carla_autonomous::StartEpisodeRequest
  :super ros::object
  :slots (_success _message ))

(defmethod carla_autonomous::StartEpisodeRequest
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

(defclass carla_autonomous::StartEpisodeResponse
  :super ros::object
  :slots ())

(defmethod carla_autonomous::StartEpisodeResponse
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

(defclass carla_autonomous::StartEpisode
  :super ros::object
  :slots ())

(setf (get carla_autonomous::StartEpisode :md5sum-) "937c9679a518e3a18d831e57125ea522")
(setf (get carla_autonomous::StartEpisode :datatype-) "carla_autonomous/StartEpisode")
(setf (get carla_autonomous::StartEpisode :request) carla_autonomous::StartEpisodeRequest)
(setf (get carla_autonomous::StartEpisode :response) carla_autonomous::StartEpisodeResponse)

(defmethod carla_autonomous::StartEpisodeRequest
  (:response () (instance carla_autonomous::StartEpisodeResponse :init)))

(setf (get carla_autonomous::StartEpisodeRequest :md5sum-) "937c9679a518e3a18d831e57125ea522")
(setf (get carla_autonomous::StartEpisodeRequest :datatype-) "carla_autonomous/StartEpisodeRequest")
(setf (get carla_autonomous::StartEpisodeRequest :definition-)
      "bool success
string message
---
")

(setf (get carla_autonomous::StartEpisodeResponse :md5sum-) "937c9679a518e3a18d831e57125ea522")
(setf (get carla_autonomous::StartEpisodeResponse :datatype-) "carla_autonomous/StartEpisodeResponse")
(setf (get carla_autonomous::StartEpisodeResponse :definition-)
      "bool success
string message
---
")



(provide :carla_autonomous/StartEpisode "937c9679a518e3a18d831e57125ea522")


