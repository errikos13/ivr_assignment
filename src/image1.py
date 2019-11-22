#!/usr/bin/env python

import sys

import cv2
import numpy as np
import roslib
import rospy
from cv_bridge import CvBridge, CvBridgeError
from numpy import cos, sin
from scipy.optimize import least_squares
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from std_msgs.msg import String

class image_converter:

  # Defines publisher and subscriber
  def __init__(self):
    self.end_effectorx = None
    self.cv_image1 = None
    # initialize the node named image_processing
    rospy.init_node('image_processing', anonymous=True)
    # initialize a publisher to send images from camera1 to a topic named image_topic1
    self.image_pub1 = rospy.Publisher("image_topic1",Image, queue_size = 1)
    # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
    #self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw",Image,self.callback1)
   # rospy.Subscriber('/target_posx', Float64, self.target_callback)
    #rospy.Subscriber('/end_effectorx', Float64, self.target_callback)
    self.joints_pub = rospy.Publisher("joints_pos",Float64MultiArray, queue_size=1)
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
    cv2.imshow('target',mask)
    cv2.waitKey(1)
    contours,hierarchy = cv2.findContours(mask, 1, 2)
    drawing = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    compactness = np.zeros(len(contours))

    for i in range(len(contours)):
        perimeter = cv2.arcLength(contours[i], 1)
        area = cv2.contourArea(contours[i])
        compactness[i] = (4*np.pi*area)/(perimeter**2)
    circle = np.argmax(compactness)
    
    
    cv2.drawContours(drawing, contours, circle, (0,255,0) )
    cv2.imshow('Contours', drawing)
    M = cv2.moments(contours[circle])
    cy = int(M['m10'] / M['m00'])
    cz = int(M['m01'] / M['m00'])
    center = self.detect_yellow(image)
    a = self.pixel2meter(image)
    cy = a*(cy - center[0])
    cz = -a*(cz - center[1])
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
      
  
  def detect_end_effector(self,image):
    a = self.pixel2meter(image)
    endPos = a * (self.detect_yellow(image) - self.detect_red(image))
    return endPos


  # Calculate the conversion from pixel to meter
  def pixel2meter(self,image):
      # Obtain the centre of each coloured blob
      circle1Pos = self.detect_blue(image)
      circle2Pos = self.detect_green(image)
      # find the distance between two circles
      dist = np.sum((circle1Pos - circle2Pos)**2)
      #return (3/np.sqrt(dist))
      return 0.03703421484500817


    
  def functions(self,t):
    endPos = self.detect_end_effector(self.cv_image1)
    a = self.pixel2meter(self.cv_image1)
    greenPos = a * (self.detect_yellow(self.cv_image1) - self.detect_green(self.cv_image1))
    f = [0,0,0,0,0]
        
    f[0] = 3*cos(t[2])*(cos(t[0])*cos(t[1]) - cos(90)*sin(t[0])*sin(t[1])) - 3*sin(t[2])*(sin(90)**2*sin(t[0]) + cos(90)*cos(t[0])*sin(t[1]) + cos(90)**2*cos(t[1])*sin(t[0])) - 2*sin(t[3])*(sin(90)*(sin(90)*cos(t[0])*sin(t[1]) - cos(90)*sin(90)*sin(t[0]) + cos(90)*sin(90)*cos(t[1])*sin(t[0])) + cos(90)*cos(t[2])*(sin(90)**2*sin(t[0]) + cos(90)*cos(t[0])*sin(t[1]) + cos(90)**2*cos(t[1])*sin(t[0])) + cos(90)*sin(t[2])*(cos(t[0])*cos(t[1]) - cos(90)*sin(t[0])*sin(t[1]))) - 2*cos(t[3])*(sin(t[2])*(sin(90)**2*sin(t[0]) + cos(90)*cos(t[0])*sin(t[1]) + cos(90)**2*cos(t[1])*sin(t[0])) - cos(t[2])*(cos(t[0])*cos(t[1]) - cos(90)*sin(t[0])*sin(t[1]))) - self.end_effectorx 
        
    f[1] = 3*sin(t[2])*(sin(90)**2*cos(t[0]) - cos(90)*sin(t[0])*sin(t[1]) + cos(90)**2*cos(t[0])*cos(t[1])) + 3*cos(t[2])*(cos(t[1])*sin(t[0]) + cos(90)*cos(t[0])*sin(t[1])) - 2*sin(t[3])*(sin(90)*(cos(90)*sin(90)*cos(t[0]) + sin(90)*sin(t[0])*sin(t[1]) - cos(90)*sin(90)*cos(t[0])*cos(t[1])) - cos(90)*cos(t[2])*(sin(90)**2*cos(t[0]) - cos(90)*sin(t[0])*sin(t[1]) + cos(90)**2*cos(t[0])*cos(t[1])) + cos(90)*sin(t[2])*(cos(t[1])*sin(t[0]) + cos(90)*cos(t[0])*sin(t[1]))) + 2*cos(t[3])*(sin(t[2])*(sin(90)**2*cos(t[0]) - cos(90)*sin(t[0])*sin(t[1]) + cos(90)**2*cos(t[0])*cos(t[1])) + cos(t[2])*(cos(t[1])*sin(t[0]) + cos(90)*cos(t[0])*sin(t[1]))) - endPos[0]
        
    f[2] = 2*cos(t[3])*(sin(t[2])*(cos(90)*sin(90) - cos(90)*sin(90)*cos(t[1])) - sin(90)*cos(t[2])*sin(t[1])) + 2*sin(t[3])*(cos(90)*cos(t[2])*(cos(90)*sin(90) - cos(90)*sin(90)*cos(t[1])) - sin(90)*(sin(90)**2*cos(t[1]) + cos(90)**2) + cos(90)*sin(90)*sin(t[1])*sin(t[2])) + 3*sin(t[2])*(cos(90)*sin(90) - cos(90)*sin(90)*cos(t[1])) - 3*sin(90)*cos(t[2])*sin(t[1]) + 2 - endPos[1]
        
    f[3] = 3*sin(t[2])*(cos(90)*sin(90) - cos(90)*sin(90)*cos(t[1])) - 3*sin(90)*cos(t[2])*sin(t[1]) + 2 - greenPos[1]
    
    f[4] = 3*sin(t[2])*(sin(90)**2*cos(t[0]) - cos(90)*sin(t[0])*sin(t[1]) + cos(90)**2*cos(t[0])*cos(t[1])) + 3*cos(t[2])*(cos(t[1])*sin(t[0]) + cos(90)*cos(t[0])*sin(t[1])) - greenPos[0]
    
     
 
        
    return f
        
        
        
        
    # Calculate the relevant joint angles from the image
  def detect_joint_angles(self,image):
    t0 = np.array([1,1,1,1])
    bl0 =-np.pi
    bu0 = np.pi
    bl1 = -(np.pi)/2
    bu1 = (np.pi)/2
    res_1 = least_squares(self.functions, t0 , bounds=([bl0, bl1, bl1, bl1], [bu0, bu1, bu1, bu1]))
    return res_1.x
 

  # Recieve data from camera 1, process it, and publish
  def callback1(self,data):
    # Recieve the image
    try:
      self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
    
    # Uncomment if you want to save the image
    #cv2.imwrite('image_copy.png', cv_image)
    im1=cv2.imshow('window1', self.cv_image1)
    cv2.waitKey(1)
    self.processing()
    
      
      
      
  def target_callback(self,data):
    # Recieve the image
    self.end_effectorx = data.data
    
    
    # Uncomment if you want to save the image
    #cv2.imwrite('image_copy.png', cv_image)
    self.processing()
    
    
   
      
  def processing(self):
    #if self.end_effectorx is not None and 
    if self.cv_image1 is not None:
    
    
   
    
        #a = self.detect_joint_angles(self.cv_image1)
        #self.joints = Float64MultiArray()
        #self.joints.data = a

        
        target_pos = self.detect_target(self.cv_image1)
        self.target_posy = Float64()
        self.target_posy.data = target_pos[0]
        
        self.target_posz = Float64()
        self.target_posz.data = target_pos[1]
        #self.target_pos = np.array(self.target_posx, self.target_posy, self.target_posz)
        
        # Publish the results
        try: 
          self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.cv_image1, "bgr8"))
         # self.joints_pub.publish(self.joints)
          self.target_posy_pub.publish(self.target_posy)
          self.target_posz_pub.publish(self.target_posz)
	    
        except CvBridgeError as e:
          print(e)
	

# call the class
def main(args):
  ic = image_converter()
  rospy.Subscriber("/camera1/robot/image_raw",Image,ic.callback1)
   # rospy.Subscriber('/target_posx', Float64, self.target_callback)
  rospy.Subscriber('/end_effectorx', Float64, ic.target_callback)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)


