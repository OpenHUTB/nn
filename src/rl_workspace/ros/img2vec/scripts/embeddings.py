#!/usr/bin/env python
import sys
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from img2vec.msg import ImgEmbedding
import std_msgs.msg


import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

 
class EmbeddingsGenerator:

  def __init__(self, camera_topic, dest_topic):
    self.image_pub = rospy.Publisher(dest_topic,ImgEmbedding, queue_size=10)
    self.image_sub = rospy.Subscriber(camera_topic, Image,self.callback)

    self.bridge = CvBridge()

    self.model = models.resnet18(pretrained=True) 
    self.layer = self.model._modules.get('avgpool')
    self.model.eval()
    
    self.preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Using standard ImageNet normalization
    ])

  def get_vector(self,image):
      image_tensor = self.preprocess(image)
      image_tensor = image_tensor.unsqueeze(0)
      my_embedding = torch.zeros(512)    # 4. Define a function that will copy the output of a layer
      def copy_data(m, i, o):
          my_embedding.copy_(o.data.reshape(o.data.size(1)))
      h = self.layer.register_forward_hook(copy_data)    # 6. Run the model on our transformed image
      self.model(image_tensor)    # 7. Detach our copy function from the layer
      h.remove()    # 8. Return the feature vector
      return my_embedding
  
  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    image_embeddings = self.get_vector(cv_image)
    rospy.logdebug("Image embedggins generated with size: %s", image_embeddings.size())
    
    # convert back to array of 64 floats
    img_embedding_np = image_embeddings.numpy()
    img_embedding_np = np.array(img_embedding_np, dtype='float64')
    rospy.logdebug(img_embedding_np[:])
    rospy.logdebug("type: %s", type(img_embedding_np))
    rospy.logdebug("shape: %s", img_embedding_np.shape)

    # generate message 
    embeddings_msg = ImgEmbedding()
    embeddings_msg.embeddings = img_embedding_np
    h = std_msgs.msg.Header()
    h.stamp = rospy.Time.now()
    embeddings_msg.header = h

    try:
      self.image_pub.publish(embeddings_msg)
    except CvBridgeError as e:
      print(e)


def main():
  rospy.init_node('EmbeddingsGenerator', anonymous=True)
  camera_topic = rospy.get_param('/img2vec/camera_topic') 
  dest_topic = rospy.get_param('/img2vec/dest_topic') 
  eg = EmbeddingsGenerator(camera_topic, dest_topic)
  rospy.loginfo("img2vec node is initialized")
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")


if __name__ == '__main__':
  main()
