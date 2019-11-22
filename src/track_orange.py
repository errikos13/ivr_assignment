#!/usr/bin/env python                                                                                                                                                                                                                    
  

import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray, Float64

from end_effector_solver import solve_joint_angles


class track_target:
    # Define subscriber
    def __init__(self):
        self.target = np.zeros(3)
        # initialize the node named image_processing
        rospy.init_node('controller', anonymous=True)
        # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
        self.target_x_sub = rospy.Subscriber('/target/x_position_controller/command', Float64, lambda x: self.point_at_target(x, None, None))
        self.target_y_sub = rospy.Subscriber('/target/y_position_controller/command', Float64, lambda y: self.point_at_target(None, y, None))
        self.target_z_sub = rospy.Subscriber('/target/z_position_controller/command', Float64, lambda z: self.point_at_target(None, None, z))
        # prepare to publish on joints
        joints = [ rospy.Publisher('/robot/joint{}_position_controller/command'.format(i), Float64, queue_size=10) for i in [1,2,3,4] ]

    def point_at_target(self, x, y, z):
        if x is not None:
            self.target[0] = x.data
        if y is not None:
            self.target[1] = y.data
        if z is not None:
            self.target[2] = z.data
        joint_angles = solve_joint_angles(self.target)
        for i in range(4):
            topic_data = Float64(joint_angles[i])
            joints[i].publish(topic_data)


# run the code if the node is called
if __name__ == '__main__':
  track_target()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
