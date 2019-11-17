#!/usr/bin/env python

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError


class image_converter:

  # Defines publisher and subscriber
  def __init__(self):
    # initialize the node named image_processing
    rospy.init_node('image_processing', anonymous=True)
    # initialize a publisher to send images from camera1 to a topic named image_topic1
    self.image_pub1 = rospy.Publisher("image_topic1",Image, queue_size = 1)
    # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
    self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw",Image,self.callback1)
    self.joints_pub = rospy.Publisher("joints_pos",Float64MultiArray, queue_size=10)
    self.target_posy_pub = rospy.Publisher("target_posy",Float64, queue_size=2)
    self.target_posz_pub = rospy.Publisher("target_posz",Float64, queue_size=2)
    # initialize the bridge between openCV and ROS
    self.bridge = CvBridge()

  def detect_red(self,image):
      # Isolate the blue colour in the image as a binary image
      mask = cv2.inRange(image, (0, 0, 100), (0, 0, 255))
      # This applies a dilate that makes the binary region larger (the more iterations the larger it becomes)
      kernel = np.ones((5, 5), np.uint8)
      mask = cv2.dilate(mask, kernel, iterations=3)
      # Obtain the moments of the binary image
      M = cv2.moments(mask)
      # Calculate pixel coordinates for the centre of the blob
      cy = int(M['m10'] / M['m00'])
      cz = int(M['m01'] / M['m00'])
      return np.array([cy, cz])
      
  def detect_target(self,image):
    mask = cv2.inRange(image, (70, 108, 128), (89, 180, 217))
    # This applies a dilate that makes the binary region larger (the more iterations the larger it becomes)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    #cv2.imshow('target',mask)
    contours,hierarchy = cv2.findContours(mask, 1, 2)
    drawing = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    compactness = np.zeros(len(contours))

    for i in range(len(contours)):
        perimeter = cv2.arcLength(contours[i], 1)
        area = cv2.contourArea(contours[i])
        compactness[i] = (4*np.pi*area)/(perimeter**2)
    circle = np.argmax(compactness)
    
    
    cv2.drawContours(drawing, contours, circle, (0,255,0) )
    #cv2.imshow('Contours', drawing)
    M = cv2.moments(contours[circle])
    cy = int(M['m10'] / M['m00'])
    cz = int(M['m01'] / M['m00'])
    center = self.detect_yellow(image)
    a = self.pixel2meter(image)
    cy = a*(cy - center[0])
    cz = a*(cz - center[1])
    return np.array([cy,cz])
    
      
    
  
 

  # Detecting the centre of the green circle
  def detect_green(self,image):
      mask = cv2.inRange(image, (0, 100, 0), (0, 255, 0))
      kernel = np.ones((5, 5), np.uint8)
      mask = cv2.dilate(mask, kernel, iterations=3)
      M = cv2.moments(mask)
      cy = int(M['m10'] / M['m00'])
      cz = int(M['m01'] / M['m00'])
      return np.array([cy, cz])


  # Detecting the centre of the blue circle
  def detect_blue(self,image):
      mask = cv2.inRange(image, (100, 0, 0), (255, 0, 0))
      kernel = np.ones((5, 5), np.uint8)
      mask = cv2.dilate(mask, kernel, iterations=3)
      M = cv2.moments(mask)
      cy = int(M['m10'] / M['m00'])
      cz = int(M['m01'] / M['m00'])
      return np.array([cy, cz])

  # Detecting the centre of the yellow circle
  def detect_yellow(self,image):
      mask = cv2.inRange(image, (0, 100, 100), (0, 255, 255))
      kernel = np.ones((5, 5), np.uint8)
      mask = cv2.dilate(mask, kernel, iterations=3)
      M = cv2.moments(mask)
      cy = int(M['m10'] / M['m00'])
      cz = int(M['m01'] / M['m00'])
      return np.array([cy, cz])


  # Calculate the conversion from pixel to meter
  def pixel2meter(self,image):
      # Obtain the centre of each coloured blob
      circle1Pos = self.detect_blue(image)
      circle2Pos = self.detect_green(image)
      # find the distance between two circles
      dist = np.sum((circle1Pos - circle2Pos)**2)
      return 3 / np.sqrt(dist)


    # Calculate the relevant joint angles from the image
  def detect_joint_angles(self,image):
    a = self.pixel2meter(image)
    # Obtain the centre of each coloured blob 
    center = a * self.detect_yellow(image)
    circle1Pos = (a * self.detect_blue(image))
    circle2Pos = (a * self.detect_green(image))
    circle3Pos = (a * self.detect_red(image))
    help1 = center - circle1Pos
    help2 = circle1Pos - circle2Pos
    help3 = circle2Pos - circle3Pos
    help4 = np.array([center[0],0])
    ja1 = np.arccos(np.dot(help1, help4)/(np.linalg.norm(help1) * np.linalg.norm(help4)))
    dot = np.dot(help1, help2)
    norm = (np.linalg.norm(help1) * np.linalg.norm(help2))
    ja2 = np.arccos(dot/norm)
    ja3 = np.arccos(np.dot(help2, help3)/(np.linalg.norm(help2) * np.linalg.norm(help3)))
    # Solve using trigonometry
    #ja1 = np.arctan2(center[0]- circle1Pos[0], center[1] - circle1Pos[1])
    #ja2 = np.arctan2(circle1Pos[0]-circle2Pos[0], circle1Pos[1]-circle2Pos[1]) - ja1
    #ja3 = np.arctan2(circle2Pos[0]-circle3Pos[0], circle2Pos[1]-circle3Pos[1]) - ja2 - ja1
    #return np.array([ja1, ja2, ja3])
    return np.array([ja1, ja2, ja3])



  # Recieve data from camera 1, process it, and publish
  def callback1(self,data):
    # Recieve the image
    try:
      self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
    
    # Uncomment if you want to save the image
    #cv2.imwrite('image_copy.png', cv_image)
    a = self.detect_joint_angles(self.cv_image1)
    self.joints = Float64MultiArray()
    self.joints.data = a
    
    target_pos = self.detect_target(self.cv_image1)
    self.target_posy = Float64()
    self.target_posy.data = target_pos[0]
    
    self.target_posz = Float64()
    self.target_posz.data = target_pos[1]
    

    #im1=cv2.imshow('window1', self.cv_image1)
    cv2.waitKey(1)
    # Publish the results
    try: 
      self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.cv_image1, "bgr8"))
      self.joints_pub.publish(self.joints)
      self.target_posy_pub.publish(self.target_posy)
      self.target_posz_pub.publish(self.target_posz)
	
    except CvBridgeError as e:
      print(e)

# call the class
def main(args):
  ic = image_converter()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)


