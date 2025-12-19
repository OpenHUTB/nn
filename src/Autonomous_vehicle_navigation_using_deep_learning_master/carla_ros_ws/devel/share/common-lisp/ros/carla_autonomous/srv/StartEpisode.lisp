; Auto-generated. Do not edit!


(cl:in-package carla_autonomous-srv)


;//! \htmlinclude StartEpisode-request.msg.html

(cl:defclass <StartEpisode-request> (roslisp-msg-protocol:ros-message)
  ((success
    :reader success
    :initarg :success
    :type cl:boolean
    :initform cl:nil)
   (message
    :reader message
    :initarg :message
    :type cl:string
    :initform ""))
)

(cl:defclass StartEpisode-request (<StartEpisode-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <StartEpisode-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'StartEpisode-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name carla_autonomous-srv:<StartEpisode-request> is deprecated: use carla_autonomous-srv:StartEpisode-request instead.")))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <StartEpisode-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader carla_autonomous-srv:success-val is deprecated.  Use carla_autonomous-srv:success instead.")
  (success m))

(cl:ensure-generic-function 'message-val :lambda-list '(m))
(cl:defmethod message-val ((m <StartEpisode-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader carla_autonomous-srv:message-val is deprecated.  Use carla_autonomous-srv:message instead.")
  (message m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <StartEpisode-request>) ostream)
  "Serializes a message object of type '<StartEpisode-request>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'message))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'message))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <StartEpisode-request>) istream)
  "Deserializes a message object of type '<StartEpisode-request>"
    (cl:setf (cl:slot-value msg 'success) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'message) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'message) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<StartEpisode-request>)))
  "Returns string type for a service object of type '<StartEpisode-request>"
  "carla_autonomous/StartEpisodeRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'StartEpisode-request)))
  "Returns string type for a service object of type 'StartEpisode-request"
  "carla_autonomous/StartEpisodeRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<StartEpisode-request>)))
  "Returns md5sum for a message object of type '<StartEpisode-request>"
  "937c9679a518e3a18d831e57125ea522")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'StartEpisode-request)))
  "Returns md5sum for a message object of type 'StartEpisode-request"
  "937c9679a518e3a18d831e57125ea522")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<StartEpisode-request>)))
  "Returns full string definition for message of type '<StartEpisode-request>"
  (cl:format cl:nil "bool success~%string message~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'StartEpisode-request)))
  "Returns full string definition for message of type 'StartEpisode-request"
  (cl:format cl:nil "bool success~%string message~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <StartEpisode-request>))
  (cl:+ 0
     1
     4 (cl:length (cl:slot-value msg 'message))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <StartEpisode-request>))
  "Converts a ROS message object to a list"
  (cl:list 'StartEpisode-request
    (cl:cons ':success (success msg))
    (cl:cons ':message (message msg))
))
;//! \htmlinclude StartEpisode-response.msg.html

(cl:defclass <StartEpisode-response> (roslisp-msg-protocol:ros-message)
  ()
)

(cl:defclass StartEpisode-response (<StartEpisode-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <StartEpisode-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'StartEpisode-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name carla_autonomous-srv:<StartEpisode-response> is deprecated: use carla_autonomous-srv:StartEpisode-response instead.")))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <StartEpisode-response>) ostream)
  "Serializes a message object of type '<StartEpisode-response>"
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <StartEpisode-response>) istream)
  "Deserializes a message object of type '<StartEpisode-response>"
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<StartEpisode-response>)))
  "Returns string type for a service object of type '<StartEpisode-response>"
  "carla_autonomous/StartEpisodeResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'StartEpisode-response)))
  "Returns string type for a service object of type 'StartEpisode-response"
  "carla_autonomous/StartEpisodeResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<StartEpisode-response>)))
  "Returns md5sum for a message object of type '<StartEpisode-response>"
  "937c9679a518e3a18d831e57125ea522")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'StartEpisode-response)))
  "Returns md5sum for a message object of type 'StartEpisode-response"
  "937c9679a518e3a18d831e57125ea522")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<StartEpisode-response>)))
  "Returns full string definition for message of type '<StartEpisode-response>"
  (cl:format cl:nil "~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'StartEpisode-response)))
  "Returns full string definition for message of type 'StartEpisode-response"
  (cl:format cl:nil "~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <StartEpisode-response>))
  (cl:+ 0
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <StartEpisode-response>))
  "Converts a ROS message object to a list"
  (cl:list 'StartEpisode-response
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'StartEpisode)))
  'StartEpisode-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'StartEpisode)))
  'StartEpisode-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'StartEpisode)))
  "Returns string type for a service object of type '<StartEpisode>"
  "carla_autonomous/StartEpisode")