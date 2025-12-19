
(cl:in-package :asdf)

(defsystem "carla_autonomous-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :std_msgs-msg
)
  :components ((:file "_package")
    (:file "Reset" :depends-on ("_package_Reset"))
    (:file "_package_Reset" :depends-on ("_package"))
    (:file "StartEpisode" :depends-on ("_package_StartEpisode"))
    (:file "_package_StartEpisode" :depends-on ("_package"))
    (:file "Stop" :depends-on ("_package_Stop"))
    (:file "_package_Stop" :depends-on ("_package"))
  ))