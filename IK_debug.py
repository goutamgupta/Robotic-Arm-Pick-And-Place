from sympy import *
from time import time
from mpmath import radians
import tf
import rospy

from kuka_arm.srv import *
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose
from mpmath import *


import numpy as np
from numpy import array 
from sympy import symbols,cos,sin,pi,simplify,sqrt,atan2,pprint,acos
from sympy.matrices import Matrix



'''
Format of test case is [ [[EE position],[EE orientation as quaternions]],[WC location],[joint angles]]
You can generate additional test cases by setting up your kuka project and running `$ roslaunch kuka_arm forward_kinematics.launch`
From here you can adjust the joint angles to find thetas, use the gripper to extract positions and orientation (in quaternion xyzw) and lastly use link 5
to find the position of the wrist center. These newly generated test cases can be added to the test_cases dictionary.
'''

test_cases = {1:[[[2.16135,-1.42635,1.55109],
                  [0.708611,0.186356,-0.157931,0.661967]],
                  [1.89451,-1.44302,1.69366],
                  [-0.65,0.45,-0.36,0.95,0.79,0.49]],
              2:[[[-0.56754,0.93663,3.0038],
                  [0.62073, 0.48318,0.38759,0.480629]],
                  [-0.638,0.64198,2.9988],
                  [-0.79,-0.11,-2.33,1.94,1.14,-3.68]],
              3:[[[-1.3863,0.02074,0.90986],
                  [0.01735,-0.2179,0.9025,0.371016]],
                  [-1.1669,-0.17989,0.85137],
                  [-2.99,-0.12,0.94,4.06,1.29,-4.12]],
              4:[[[-0.0041726,0.34072,3.5038],
                  [0.85348,0.44389,0.26409,-0.0692427]],
                  [-0.12229,0.23472,3.76372],
                  [2.44,-0.42,-0.95,4.71,1.02,-3.04]],
              5:[[[-0.13217,-1.5285,2.92805],
                  [-0.83251,0.081294,-0.13317,0.531592]],
                  [-0.215395,-1.779108,3.054832],
                  [-1.85,1.03,-2.59,4.71,-1.44,1.14]]}

def test_code(test_case):
    ## Set up code
    ## Do not modify!
    x = 0
    class Position:
        def __init__(self,EE_pos):
            self.x = EE_pos[0]
            self.y = EE_pos[1]
            self.z = EE_pos[2]
    class Orientation:
        def __init__(self,EE_ori):
            self.x = EE_ori[0]
            self.y = EE_ori[1]
            self.z = EE_ori[2]
            self.w = EE_ori[3]

    position = Position(test_case[0][0])
    orientation = Orientation(test_case[0][1])

    class Combine:
        def __init__(self,position,orientation):
            self.position = position
            self.orientation = orientation
    comb = Combine(position,orientation)

    class Pose:
        def __init__(self,comb):
            self.poses = [comb]

    req = Pose(comb)
    start_time = time()
    
    ########################################################################################
    ## 
    
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



    ## Insert IK code here!

    #get position and orientation from test case
    px = req.poses[x].position.x
    py = req.poses[x].position.y
    pz = req.poses[x].position.z
    
    (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
        [req.poses[x].orientation.x, req.poses[x].orientation.y,
        req.poses[x].orientation.z, req.poses[x].orientation.w])

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
    
    
    ## 
    ########################################################################################
    
    
    ########################################################################################
    ## For additional debugging add your forward kinematics here. Use your previously calculated thetas
    ## as the input and output the position of your end effector as your_ee = [x,y,z]

    ## (OPTIONAL) YOUR CODE HERE!

    ### Your FK code here
    #Total Homogeneous transform between base link and gripper link taking into account the corrction Matrix 

    #T_total = simplify(T0_EE * T_corr) 
    # End effector position calculation based on forward kinematics 
    # print("T0_Total = ",T_total.evalf(subs={q1: 0, q2: 0, q3: 0, q4: 0, q5: 0, q6: 0}))

    T_EE = T_total.evalf(subs={q1: theta1, q2: theta2, q3: theta3, q4: theta4, q5: theta5, q6: theta6})
    print("T_EE",T_EE)
    # End effector position co-ordinates from forward kinematics
    #ee_x_f = T_total[0,3].evalf(subs={q1: theta1, q2: theta2, q3: theta3, q4: theta4, q5: theta5, q6: theta6})
    #ee_y_f = T_total[1,3].evalf(subs={q1: theta1, q2: theta2, q3: theta3, q4: theta4, q5: theta5, q6: theta6})
    #ee_z_f = T_total[2,3].evalf(subs={q1: theta1, q2: theta2, q3: theta3, q4: theta4, q5: theta5, q6: theta6})




    ## End your code input for forward kinematics here!
    ########################################################################################

    ## For error analysis please set the following variables of your WC location and EE location in the format of [x,y,z]

    ## For error analysis please set the following variables of your WC location and EE location in the format of [x,y,z]
    your_wc = [WC_x,WC_y,WC_z] # <--- Load your calculated WC values in this array
    your_ee = [T_EE[0,3],T_EE[1,3],T_EE[2,3]] # <--- Load your calculated end effector value from your forward kinematics
    ########################################################################################

    ## Error analysis
    print ("\nTotal run time to calculate joint angles from pose is %04.4f seconds" % (time()-start_time))

    # Find WC error
    if not(sum(your_wc)==3):
        wc_x_e = abs(your_wc[0]-test_case[1][0])
        wc_y_e = abs(your_wc[1]-test_case[1][1])
        wc_z_e = abs(your_wc[2]-test_case[1][2])
        wc_offset = sqrt(wc_x_e**2 + wc_y_e**2 + wc_z_e**2)
        print ("\nWrist error for x position is: %04.8f" % wc_x_e)
        print ("Wrist error for y position is: %04.8f" % wc_y_e)
        print ("Wrist error for z position is: %04.8f" % wc_z_e)
        print ("Overall wrist offset is: %04.8f units" % wc_offset)

    # Find theta errors
    t_1_e = abs(theta1-test_case[2][0])
    t_2_e = abs(theta2-test_case[2][1])
    t_3_e = abs(theta3-test_case[2][2])
    t_4_e = abs(theta4-test_case[2][3])
    t_5_e = abs(theta5-test_case[2][4])
    t_6_e = abs(theta6-test_case[2][5])
    print ("\nTheta 1 error is: %04.8f" % t_1_e)
    print ("Theta 2 error is: %04.8f" % t_2_e)
    print ("Theta 3 error is: %04.8f" % t_3_e)
    print ("Theta 4 error is: %04.8f" % t_4_e)
    print ("Theta 5 error is: %04.8f" % t_5_e)
    print ("Theta 6 error is: %04.8f" % t_6_e)
    print ("\n**These theta errors may not be a correct representation of your code, due to the fact \
           \nthat the arm can have muliple positions. It is best to add your forward kinmeatics to \
           \nconfirm whether your code is working or not**")
    print (" ")

    # Find FK EE error
    if not(sum(your_ee)==3):
        ee_x_e = abs(your_ee[0]-test_case[0][0][0])
        ee_y_e = abs(your_ee[1]-test_case[0][0][1])
        ee_z_e = abs(your_ee[2]-test_case[0][0][2])
        ee_offset = sqrt(ee_x_e**2 + ee_y_e**2 + ee_z_e**2)
        print ("\nEnd effector error for x position is: %04.8f" % ee_x_e)
        print ("End effector error for y position is: %04.8f" % ee_y_e)
        print ("End effector error for z position is: %04.8f" % ee_z_e)
        print ("Overall end effector offset is: %04.8f units \n" % ee_offset)




if __name__ == "__main__":
    # Change test case number for different scenarios
    test_case_number = 5

    test_code(test_cases[test_case_number])





