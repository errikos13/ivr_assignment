#!/usr/bin/env python

import sys

import cv2
import numpy as np
import roslib
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from std_msgs.msg import String

class image_converter:

	# Defines publisher and subscriber
    def __init__(self):
        # initialize the node named image_processing
        rospy.init_node('image_processing', anonymous=True)
        # initialize a publisher to send images from camera2 to a topic named image_topic2
        self.image_pub2 = rospy.Publisher("image_topic2",Image, queue_size = 1)
        self.target_posx_pub = rospy.Publisher("target_posx",Float64, queue_size=2)
        self.end_effectorx_pub = rospy.Publisher("end_effectorx", Float64, queue_size = 2)
        # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
        self.image_sub2 = rospy.Subscriber("/camera2/robot/image_raw",Image,self.callback2)
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
        cx = int(M['m10'] / M['m00'])
        cz = int(M['m01'] / M['m00'])
        return np.array([cx, cz])


    # Detecting the centre of the green circle
    def detect_green(self,image):
        mask = cv2.inRange(image, (0, 100, 0), (0, 255, 0))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        M = cv2.moments(mask)
        cx = int(M['m10'] / M['m00'])
        cz = int(M['m01'] / M['m00'])
        return np.array([cx, cz])


    # Detecting the centre of the blue circle
    def detect_blue(self,image):
        mask = cv2.inRange(image, (100, 0, 0), (255, 0, 0))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        M = cv2.moments(mask)
        cx = int(M['m10'] / M['m00'])
        cz = int(M['m01'] / M['m00'])
        return np.array([cx, cz])

    # Detecting the centre of the yellow circle
    def detect_yellow(self,image):
        mask = cv2.inRange(image, (0, 100, 100), (0, 255, 255))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        M = cv2.moments(mask)
        cx = int(M['m10'] / M['m00'])
        cz = int(M['m01'] / M['m00'])
        return np.array([cx, cz])

    def detect_end_effector(self,image):
        a = self.pixel2meter(image)
        endPos = a * (self.detect_yellow(image) - self.detect_red(image))
        return endPos


    def detect_target(self,image):
        mask = cv2.inRange(image, (70, 108, 128), (89, 180, 217))
        # This applies a dilate that makes the binary region larger (the more iterations the larger it becomes)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
       # cv2.imshow('target',mask)
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
        cx = a*(cy - center[0])
        cz = a*(cz - center[1])
        return np.array([cx,cz])


    # Calculate the conversion from pixel to meter
    def pixel2meter(self,image):
        # Obtain the centre of each coloured blob
        circle1Pos = self.detect_blue(image)
        circle2Pos = self.detect_green(image)
        # find the distance between two circles
        dist = np.sum((circle1Pos - circle2Pos)**2)
        #return 3 / np.sqrt(dist)
        return 0.03703421484500817



    # Recieve data, process it, and publish
    def callback2(self,data):
        # Recieve the image
        try:
            self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        # Uncomment if you want to save the image
        #cv2.imwrite('image_copy.png', cv_image)
        #im2=cv2.imshow('window2', self.cv_image2)
        cv2.waitKey(1)

        target_pos = self.detect_target(self.cv_image2)
        endPos = self.detect_end_effector(self.cv_image2)
        self.end_effectorx = Float64()
        self.end_effectorx.data = endPos[0]
        self.target_posx = Float64()
        self.target_posx.data = target_pos[0]


# Publish the results
        try:
            self.image_pub2.publish(self.bridge.cv2_to_imgmsg(self.cv_image2, "bgr8"))
            self.target_posx_pub.publish(self.target_posx)
            self.end_effectorx_pub.publish(self.end_effectorx)
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

