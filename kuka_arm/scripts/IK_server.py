#!/usr/bin/env python

# Copyright (C) 2017 Udacity Inc.
#
# This file is part of Robotic Arm: Pick and Place project for Udacity
# Robotics nano-degree program
#
# All Rights Reserved.

# Author: Goutam Gupta

# import modules
import rospy
import tf
from kuka_arm.srv import *
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose
from mpmath import *
from sympy import *
from sympy.matrices import Matrix
import numpy as np
from numpy import array 
from sympy import symbols,cos,sin,pi,simplify,sqrt,atan2,pprint,acos


def handle_calculate_IK(req):
    rospy.loginfo("Received %s eef-poses from the plan" % len(req.poses))
    if len(req.poses) < 1:
        print "No valid poses received"
        return -1
    else:

        ### Your FK code here
          # Create symbols
        q1,q2,q3,q4,q5,q6,q7 = symbols('q1:8')
        d1,d2,d3,d4,d5,d6,d7 = symbols('d1:8')
        a0,a1,a2,a3,a4,a5,a6 = symbols('a0:7') 
        alpha0,alpha1,alpha2,alpha3,alpha4,alpha5,alpha6 = symbols('alpha0:7')
        #
        # Create Modified DH parameters
        DH_Table = {
          alpha0 : 0, a0:0, d1:0.75, q1:q1,
          alpha1 : -pi/2., a1:0.35, d2:0, q2:q2-pi/2.,
          alpha2 : 0, a2:1.25, d3:0, q3:q3,
          alpha3 : -pi/2., a3:-0.054, d4:1.5, q4:q4,
          alpha4 : pi/2., a4:0, d5:0, q5:q5,
          alpha5 : -pi/2., a5:0, d6:0, q6:q6,
          alpha6 : 0, a6:0, d7:0.303, q7:0, 
           } 

        def TF_Matrix ( alpha,a,d,q) :
            TF = Matrix( [ [cos(q),-sin(q),0,a],
                        [sin(q)*cos(alpha),cos(q)*cos(alpha),-sin(alpha),-sin(alpha)*d],
                        [sin(q)*sin(alpha),cos(q)*sin(alpha), cos(alpha), cos (alpha)*d],
                        [0,0,0,1]])
            return TF 
        #
        # Create individual transformation matrices
        T0_1 = TF_Matrix(alpha0,a0,d1,q1).subs(DH_Table) 
        T1_2 = TF_Matrix(alpha1,a1,d2,q2).subs(DH_Table)
        T2_3 = TF_Matrix(alpha2,a2,d3,q3).subs(DH_Table)
        T3_4 = TF_Matrix(alpha3,a3,d4,q4).subs(DH_Table) 
        T4_5 = TF_Matrix(alpha4,a4,d5,q5).subs(DH_Table)
        T5_6 = TF_Matrix(alpha5,a5,d6,q6).subs(DH_Table)
        T6_EE =TF_Matrix(alpha6,a6,d7,q7).subs(DH_Table)

        # Composition of Homogeneous transforms 
        T0_2  = simplify(T0_1 * T1_2)    #base link to link 2 
        T0_3  = simplify(T0_2 * T2_3)    #base link to link 3
        T0_4  = simplify(T0_3 * T3_4)    #base link to link 4
        T0_5  = simplify(T0_4 * T4_5)    #base link to link 5
        T0_6  = simplify(T0_5 * T5_6)    #base link to link 6
        T0_EE = simplify(T0_6 * T6_EE)  #base link to End Effector 

        # correction Matrix to account for orientation difference between DH co$
        T_z = Matrix([[cos(np.pi), -sin(np.pi), 0, 0],
                     [sin(np.pi),  cos(np.pi), 0, 0],
                     [0,             0,        1., 0],
                     [0,             0,        0, 1.]])

        T_y = Matrix([[cos(-np.pi/2.), 0, sin(-np.pi/2.), 0],
                      [0,             1., 0,             0],
                      [-sin(-np.pi/2.),0, cos(-np.pi/2.), 0],
                      [0,             0,        0,      1]])



        T_corr = simplify(T_z * T_y) 
        R_corr = T_corr[0:3,0:3]

        #Total Homogeneous transform between base link and gripper link 
        T_total = simplify(T0_EE * T_corr)      

        print("T_Total = ",T_total.evalf(subs={q1: 0.7, q2: 0.5, q3: 0.6, q4: 0, q5: 0, q6: 0}))

        ### define function for getting rotation Matrix from euler angle    
        def rotation_from_euler(r,p,y):
                  R_z_y = Matrix([[cos(y),-sin(y),0],[sin(y),cos(y),0],[0,0,1]])
                  R_y_p = Matrix([[cos(p),0,sin(p)],[0,1,0],[-sin(p),0,cos(p)]])
                  R_x_r = Matrix([[1,0,0],[0,cos(r),-sin(r)],[0,sin(r),cos(r)]])
                  Rot = R_z_y * R_y_p *R_x_r 
                  return Rot

        # Initialize service response
        joint_trajectory_list = []
        for x in xrange(0, len(req.poses)):
            # IK code starts here
            joint_trajectory_point = JointTrajectoryPoint()

            # Extract end-effector position and orientation from request
            # px,py,pz = end-effector position
            # roll, pitch, yaw = end-effector orientation
            px = req.poses[x].position.x
            py = req.poses[x].position.y
            pz = req.poses[x].position.z

            (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
                [req.poses[x].orientation.x, req.poses[x].orientation.y,
                    req.poses[x].orientation.z, req.poses[x].orientation.w])

            ### Your IK code here
            #Calculate rotation matrix from roll,pitch and yaw values
            RTOT_EE = rotation_from_euler(roll,pitch,yaw)
    
    
            # Compensate for rotation discrepancy between DH parameters and Gazebo
            R_corr = T_corr[0:3,0:3]
            RTOT_EE = RTOT_EE * R_corr

            #calculation of wrist centre co-rdinates
    
            WC_x = px-0.303*RTOT_EE[0,2]
            WC_y = py-0.303*RTOT_EE[1,2]
            WC_z = pz-0.303*RTOT_EE[2,2]

            #calculation of theta1 based on the triangle shown in write up
            theta1= atan2(WC_y,WC_x)
            #print("theta1 = ",theta1.evalf(subs={q1: 1.5, q2: 2.5, q3: 3.0, q4: 3.14, q5: 2.14, q6: 1.14}))
    

            #calculation of theta2 and theta3 based on the triangle mentioned in the write up
            #calculation of sides of the triangle formed by joint 2,joint 3 and the wrist centre
            C=1.25
            A=round(sqrt(1.5**2+0.054**2),7)
            B=sqrt((sqrt(WC_x**2 + WC_y**2) -0.35)**2 + (WC_z-0.75)**2)
            cos_a=(B**2+C**2-A**2)/(2*B*C)
            cos_b=(A**2+C**2-B**2)/(2*A*C)
            angle_a = acos(cos_a)
            angle_b = acos(cos_b)
            theta2 = pi/2 - atan2((WC_z-0.75),(sqrt(WC_x**2+WC_y**2)-0.35)) - angle_a
            theta3 = pi/2 - angle_b - atan2(0.054,1.5)    
   

            #calculation of theta4,theta5 and theta 6 based on the euler angle concept for a spherical wrist :joint4,joint5,joint6
            R0_3 = T0_3[0:3,0:3]
            R0_3 = R0_3.evalf(subs={q1: theta1, q2: theta2, q3: theta3})
            #print(R0_3)
            R3_6 = R0_3.inv("LU") * RTOT_EE
    
            #extracting all elements of R3_6 for theta4,theta5 and theta6 calculations
            r11 = R3_6[0,0]
            r12 = R3_6[0,1]
            r13 = R3_6[0,2]
            r21 = R3_6[1,0]
            r22 = R3_6[1,1]
            r23 = R3_6[1,2]
            r31 = R3_6[2,0]
            r32 = R3_6[2,1]
            r33 = R3_6[2,2]

            theta4 = atan2(R3_6[2,2],-R3_6[0,2])
            theta5 = atan2(sqrt(R3_6[0,2]**2+R3_6[2,2]**2),R3_6[1,2])
            theta6 = atan2(-R3_6[1,1],R3_6[1,0])

            # Populate response for the IK request
            # In the next line replace theta1,theta2...,theta6 by your joint angle variables
            joint_trajectory_point.positions = [theta1, theta2, theta3, theta4, theta5, theta6]
            joint_trajectory_list.append(joint_trajectory_point)

        rospy.loginfo("length of Joint Trajectory List: %s" % len(joint_trajectory_list))

        return CalculateIKResponse(joint_trajectory_list)


def IK_server():
    # initialize node and declare calculate_ik service
    rospy.init_node('IK_server')
    s = rospy.Service('calculate_ik', CalculateIK, handle_calculate_IK)
    print "Ready to receive an IK request"
    rospy.spin()

if __name__ == "__main__":
    IK_server()





