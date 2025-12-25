
(cl:in-package :asdf)

(defsystem "carla_tracking-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "TrackedObject" :depends-on ("_package_TrackedObject"))
    (:file "_package_TrackedObject" :depends-on ("_package"))
  ))