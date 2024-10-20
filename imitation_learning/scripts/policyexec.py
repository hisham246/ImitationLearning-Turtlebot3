#!/usr/bin/env python

# This script performs an execution of the best policy to get from start
# to goal point. The program will reset the robot to the start state once
# it reaches the goal before quitting itself. Manual restart will be needed
# if the robot crashes into the wall during execution (highly unlikely!! :D)


import rospy
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool

import numpy as np
import json

import argparse
import os
from rospkg import RosPack
import random

from model_il import ImitationNet
import model_il
import torch


# global data msgs
expert_cmd = None
laser = None 
pos = None
true_pos = None



# Topic Callbacks
def expertCmdCallback(vel):
	global expert_cmd
	expert_cmd = vel

def laserCallback(scan):
	global laser
	laser = scan 

def amclCallback(amcl):
	global pos
	pos = amcl

def odomCallback(odom):
	global true_pos
	true_pos = odom

def resetRobotState():
	pub_reset.publish(True)
	rospy.sleep(0.1)
	pub_reset.publish(True)		# Repeat to make sure reset at the right pose

def execute():
	global model, pos, laser

	# Load model
	if os.path.exists(args.model_dir + 'dagger/model.pt'):
		model.load(args.model_dir + 'dagger/model.pt')
	model = model.to(args.device)
	model.eval()

	# Initialize rate for controlling robot and updating observations
	rate = rospy.Rate(20) #Hz
	#Initialize robot command
	robot_cmd = Twist()

	print('Starting policy execution.')

	# Keep executing until goal is reached
	while True:
		# Run network policy to get control command 
		odom_input = np.array([[pos.pose.pose.position.x, 
								pos.pose.pose.position.y,
								pos.pose.pose.position.z,
								pos.pose.pose.orientation.x, 
								pos.pose.pose.orientation.y, 
								pos.pose.pose.orientation.z, 
								pos.pose.pose.orientation.w]])
		laser_scan = np.array([laser.ranges])

		vel = model_il.test(model, odom_input, laser_scan)
		robot_cmd.linear.x = vel[0,0]
		robot_cmd.angular.z = vel[0,1]
		pub_cmd.publish(robot_cmd)

		if abs(pos.pose.pose.position.x - 6.0) < 1.0 and \
				abs(pos.pose.pose.position.y + 1.0) < 0.3:
			# Stop the robot
			robot_cmd.linear.x, robot_cmd.angular.z = 0.0, 0.0
			pub_cmd.publish(robot_cmd)
			# Reset robot to start state
			print("Goal is reached. Resetting robot to start")
			resetRobotState()
			print('Policy execution successful. Quitting Program.')
			break

		# Sleep for the remaining time
		rate.sleep()



if __name__=='__main__':
	os.chdir(RosPack().get_path('imitation_learning'))
	parser = argparse.ArgumentParser(description="parse args")
	parser.add_argument('--model_dir', type=str, 
						default=os.getcwd() + '/models/')
	parser.add_argument('--device', type=str, default='cpu')
	args = parser.parse_args()

	# configure cuda
	if torch.cuda.is_available():
	    args.device = torch.device('cuda')
	else:
	    args.device = "cpu"

	rospy.init_node('policyexec', anonymous=True)

	# add ROS subscribers and publishers
	rospy.Subscriber("scan", LaserScan, laserCallback)
	rospy.Subscriber("odom", Odometry, odomCallback)
	rospy.Subscriber("amcl_pose", PoseWithCovarianceStamped, amclCallback)
	rospy.Subscriber("/cmd_vel_temp", Twist, expertCmdCallback)
	pub_reset = rospy.Publisher("reset_robot_pos", Bool, queue_size=10)
	pub_cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

	# Initialize model
	model = ImitationNet(control_dim=2, device=args.device)

	# Execute the policy
	execute()