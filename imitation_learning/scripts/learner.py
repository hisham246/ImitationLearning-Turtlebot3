#!/usr/bin/env python

# This script contains the standard supervised learning algorithm. It 
# repeatedly collects episodes of expert demonstrations using the expert
# policy with some additive noise and trains the neural network policy
# with the collected dataset at the end. It also contains a separate 
# execution feature that tests the current model in getting the robot from 
# point A to point B.


import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Twist, Pose, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String, Bool
from imitation_learning.srv import Mode, ModeResponse

import numpy as np
import json

import argparse
import os
from rospkg import RosPack
import random

import sys
sys.path.append('/home/hisham246/uwaterloo/ME780/turtlebot_ws/src/ImitationLearning-Turtlebot3/imitation_learning/scripts')

from model_il import ImitationNet
import model_il
import torch


# constants for the mode
IDLE = 0
COLLECT = 1
TRAIN = 2
EXECUTE = 3


class Dagger():
	def __init__(self, args):
		# miscellaneous variables
		self.robot_cmd = Twist()
		self.expert_cmd = None
		self.laser = None 
		self.pos = None
		self.true_pos = None
		self.currmode = None
		self.num_traj = 0
		self.device = args.device
		self.model_dir = args.model_dir
		self.data_dir = args.data_dir
		# Initialize dataset and sliding window for data recording
		self.D = None
		self.clearDataset()

		# Initialize expert goal
		self.goal = MoveBaseGoal()
		self.goal.target_pose.header.frame_id = "map"
		self.goal.target_pose.header.stamp = rospy.Time.now()
		self.goal.target_pose.pose.position.x = 6.0
		self.goal.target_pose.pose.position.y = -1.0
		self.goal.target_pose.pose.orientation.z = 0.7071
		self.goal.target_pose.pose.orientation.w = -0.7071

		# add ROS subscribers
		rospy.Subscriber("scan", LaserScan, self.laserCallback)
		rospy.Subscriber("odom", Odometry, self.odomCallback)
		rospy.Subscriber("amcl_pose", PoseWithCovarianceStamped, self.amclCallback)
		rospy.Subscriber("/cmd_vel_temp", Twist, self.expertCmdCallback)

		# add ROS publishers
		self.pub_reset = rospy.Publisher("reset_robot_pos", Bool, queue_size=10)
		self.pub_cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

		# service handler
		self.mode_service = rospy.Service('mode_change', Mode, self.handleModeChange)

		# Initialize move_base SimpleActionClient
		self.client = actionlib.SimpleActionClient('move_base',MoveBaseAction)
		self.client.wait_for_server()

		# Initialize model
		self.model = ImitationNet(control_dim=2, device=args.device)


	# Topic Callbacks
	def expertCmdCallback(self, vel):
		self.expert_cmd = vel

	def laserCallback(self, scan):
		self.laser = scan 

	def amclCallback(self, amcl):
		self.pos = amcl

	def odomCallback(self, odom):
		self.true_pos = odom

	# Service Callback
	def handleModeChange(self, req):
		reqmode = int(req.reqmode)
		if reqmode == COLLECT:
			rospy.loginfo(rospy.get_caller_id() + ": COLLECT MODE")
		elif reqmode == TRAIN:
			rospy.loginfo(rospy.get_caller_id() + ": TRAIN MODE")
		elif reqmode == EXECUTE:
			rospy.loginfo(rospy.get_caller_id() + ": EXECUTE MODE")
		elif reqmode == CLEARDATA:
			rospy.loginfo(rospy.get_caller_id() + ": CLEARDATA MODE")
		elif reqmode == IDLE:
			self.pos = self.laser = None 
			rospy.loginfo(rospy.get_caller_id() + ": IDLE MODE")
		else:
			reqmode = self.currmode 
			
		self.currmode = reqmode
		return ModeResponse(self.currmode)

	def recordData2File(self):
		# Write to the correct file name
		file_name = self.data_dir+"trajectory_"+str(self.num_traj)+'.json'
		while os.path.exists(file_name):
			self.num_traj += 1
			file_name = self.data_dir+"trajectory_"+str(self.num_traj)+'.json'

		# Write to json file
		if len(self.D["robot_vel"]) > 0:
			with open(file_name, 'w') as fout:
				json.dump(self.D, fout)
			self.num_traj += 1

	def resetRobotState(self):
		self.pub_reset.publish(True)
		rospy.sleep(0.1)
		self.pub_reset.publish(True)		# Repeat to make sure reset at the right pose

	def clearDataset(self):
		# Initialize/Reset Dataset D
		self.D = {'robot_pos': [], 'laser_scan': [], 'robot_vel':[]}


	def collect(self):
		# Noise variable
		noise_count = 0
		noise = 0.0

		# Start the expert planning first and let it run in the background
		self.client.send_goal(self.goal)
		print('starting expert planning')

		# Control robot until robot reached goal state
		while True:
			if self.expert_cmd is not None:
			# Check if goal is reached
				if (abs(self.true_pos.pose.pose.position.x - 6.0) < 0.5 and \
					abs(self.true_pos.pose.pose.position.y + 1.0) < 0.1):
					# Stop the robot
					self.robot_cmd.linear.x, self.robot_cmd.angular.z = 0.0, 0.0
					self.pub_cmd.publish(self.robot_cmd)
					# Stop the expert policy path planning
					print('Stopping expert planning.')
					self.client.cancel_goal()
					# Reset robot to start state
					print("Demonstration complete. Resetting robot to start")
					self.resetRobotState()
					# Change mode back to IDLE
					print('Collection done.')
					self.currmode = IDLE
					break

			
				# Always use expert policy
				print('expert {:10.6f}, {:10.6f}'.format(self.expert_cmd.linear.x, self.expert_cmd.angular.z))
				self.robot_cmd.linear.x = self.expert_cmd.linear.x
				if noise_count % 100 == 0:
					if np.random.randint(2):
						noise = np.random.uniform(low=0.1, high=0.2)
					else:
						noise = np.random.uniform(low=-0.2, high=-0.1)
				self.robot_cmd.angular.z = self.expert_cmd.angular.z + noise
				self.pub_cmd.publish(self.robot_cmd)

				# Update D
				robot_pos = [self.pos.pose.pose.position.x, self.pos.pose.pose.position.y,
							self.pos.pose.pose.position.z, self.pos.pose.pose.orientation.x,
							self.pos.pose.pose.orientation.y, self.pos.pose.pose.orientation.z, 
							self.pos.pose.pose.orientation.w]
				robot_vel = [self.expert_cmd.linear.x, self.expert_cmd.angular.z]

				self.D["robot_pos"] += [robot_pos]
				self.D["laser_scan"] += [self.laser.ranges]
				self.D["robot_vel"] += [robot_vel]

				noise_count += 1

			# Sleep for the remaining time
			rate.sleep()


	def train(self):
		# Load model again
		if os.path.exists(self.model_dir + 'model.pt'):
			self.model.load(self.model_dir + 'model.pt')
		self.model = self.model.to(self.device)

		# train neural network
		self.model = model_il.train(self.model, mode='supervised')
		self.model.eval()

		# Reset robot to start state
		print("Training round complete. Resetting robot to start")
		self.resetRobotState()

		# Done training, return to idle
		print("Returning to Idle mode.")
		self.currmode=IDLE

	def execute(self):
		# Load model again
		if os.path.exists(self.model_dir + 'model.pt'):
			self.model.load(self.model_dir + 'model.pt')
		self.model = self.model.to(self.device)
		self.model.eval()

		# Keep executing until goal is reached
		while True:
			# Run network policy to get control command 
			odom_input = np.array([[self.pos.pose.pose.position.x, 
									self.pos.pose.pose.position.y,
									self.pos.pose.pose.position.z,
									self.pos.pose.pose.orientation.x, 
									self.pos.pose.pose.orientation.y, 
									self.pos.pose.pose.orientation.z, 
									self.pos.pose.pose.orientation.w]])
			laser_scan = np.array([self.laser.ranges])

			vel = model_il.test(self.model, odom_input, laser_scan)
			self.robot_cmd.linear.x = vel[0,0]
			self.robot_cmd.angular.z = vel[0,1]
			self.pub_cmd.publish(self.robot_cmd)

			if abs(self.true_pos.pose.pose.position.x - 6.0) < 1.0 and \
					abs(self.true_pos.pose.pose.position.y + 1.0) < 0.3:
				# Stop the robot
				self.robot_cmd.linear.x, self.robot_cmd.angular.z = 0.0, 0.0
				self.pub_cmd.publish(self.robot_cmd)
				# Reset robot to start state
				print("Goal is reached. Resetting robot to start")
				self.resetRobotState()
				# Change mode back to IDLE
				print('Testing round successful. Going back to IDLE mode.')
				self.currmode = IDLE
				break

			# Sleep for the remaining time
			rate.sleep()

		#done training, return to idle
		self.currmode=IDLE



if __name__=='__main__':
	os.chdir(RosPack().get_path('imitation_learning'))
	parser = argparse.ArgumentParser(description="parse args")
	parser.add_argument('--model_dir', type=str, 
						default=os.getcwd() + '/models/supervised/')
	parser.add_argument('--data_dir', type=str, 
						default=os.getcwd() + '/data/supervised/')
	parser.add_argument('--device', type=str, default='cpu')
	args = parser.parse_args()

	# configure cuda
	if torch.cuda.is_available():
	    args.device = torch.device('cuda')
	else:
	    args.device = "cpu"

	rospy.init_node('learner', anonymous=True)

	dagger = Dagger(args)

	# Initialize rate for controlling robot and updating observations
	rate = rospy.Rate(20) #Hz

	# main loop
	while not rospy.is_shutdown():
		if dagger.currmode == COLLECT:
			dagger.collect()
		elif dagger.currmode == TRAIN:
			dagger.train()
		elif dagger.currmode == EXECUTE:
			dagger.execute()
		else:
			# write data to json file and reset D
			if len(dagger.D['robot_pos']) > 0:
				dagger.recordData2File()
				dagger.clearDataset()
				print('Going back to collect')
				dagger.expert_cmd = None
				dagger.currmode = COLLECT
			dagger.clearDataset()

		# sleep for the remaining time
		rate.sleep()