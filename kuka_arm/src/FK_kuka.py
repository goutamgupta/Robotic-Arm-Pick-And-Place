#!/usr/bin/env python

# Copyright (C) 2017 Udacity Inc.
#
# This file is part of Robotic Arm: Pick and Place project for Udacity
# Robotics nano-degree program
#
# All Rights Reserved.

# Author: Goutam Gupta

# import modules
import numpy as np
from numpy import array 
from sympy import symbols,cos,sin,pi,simplify,sqrt,atan2,acos
from sympy.matrices import Matrix
from sympy import *
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
        #
        # Define Modified DH Transformation matrix
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

        # correction Matrix to account for orientation difference between DH convention and end effector frame
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



        #Numerically Evaluate Each Transforms 
#print("T0_1 = ", T0_1.evalf(subs={q1: 0, q2: 0, q3: 0, q4: 0, q5: 0, q6: 0}))
#print("T0_2 = ", T0_2.evalf(subs={q1: 0, q2: 0, q3: 0, q4: 0, q5: 0, q6: 0}))
#print("T0_3 = ", T0_3.evalf(subs={q1: 0, q2: 0, q3: 0, q4: 0, q5: 0, q6: 0}))
#print("T0_4 = ", T0_4.evalf(subs={q1: 0, q2: 0, q3: 0, q4: 0, q5: 0, q6: 0}))
#print("T0_5 = ", T0_5.evalf(subs={q1: 0, q2: 0, q3: 0, q4: 0, q5: 0, q6: 0}))
#print("T0_6 = ", T0_6.evalf(subs={q1: 0, q2: 0, q3: 0, q4: 0, q5: 0, q6: 0}))
#print("T0_EE = ",T0_EE.evalf(subs={q1: 0, q2: 0, q3: 0, q4: 0, q5: 0, q6: 0}))
print("T_Total = ",T_total.evalf(subs={q1: 0.7, q2: 0.5, q3: 0.6, q4: 0, q5: 0, q6: 0}))

        # Extract rotation matrices from the transformation matrices


R_total = T_total[0:3,0:3]



#print("R_Total = ",R_total.evalf(subs={q1: 0.7, q2: 0.5, q3: 0.6, q4: 0, q5: 0, q6: 0}))

# End effector position x,y,z cordinates
P_EE= T_total[0:3,3]

#
R0_6 = T0_6[0:3,0:3]
ROT_EE = R0_6*R_corr

#print("P_EE = ",P_EE.evalf(subs={q1: 0.7, q2: 0.5, q3: 0.6, q4: 0, q5: 0, q6: 0}))
#print("ROT_EE = ",ROT_EE.evalf(subs={q1: 0.7, q2: 0.5, q3: 0.6, q4: 0, q5: 0, q6: 0}))

#WC = P_EE - 0.303*.ROT_EE

#print("WC = ",WC.evalf(subs={q1: 0.5, q2: 0, q3: 0, q4: 0, q5: 0, q6: 0}))

WC_x = P_EE[0,0]-0.303*ROT_EE[0,2]
WC_y = P_EE[1,0]-0.303*ROT_EE[1,2]
WC_z = P_EE[2,0]-0.303*ROT_EE[2,2]
WC=Matrix([[WC_x],[WC_y],[WC_z]])

theta1= atan2(WC_y,WC_x)
   
print("WC = ",WC.evalf(subs={q1: -1.85, q2: 1.03, q3: -2.59, q4: 4.71, q5: -1.44, q6: 1.14}))
print("theta1 = ",theta1.evalf(subs={q1: 1.5, q2: 2.5, q3: 3.0, q4: 3.14, q5: 2.14, q6: 1.14}))

#print("theta1 = ",theta1)

C=1.25
A=round(sqrt(1.5**2+0.054**2),7)
B=sqrt((sqrt(WC_x**2 + WC_y**2) -0.35)**2 + (WC_z-0.75)**2)


cos_a=(B**2+C**2-A**2)/(2*B*C)
cos_b=(A**2+C**2-B**2)/(2*A*C)

angle_a = acos(cos_a)
angle_b = acos(cos_b)
theta2 = pi/2 - atan2((WC_z-0.75),(sqrt(WC_x**2+WC_y**2)-0.35)) - angle_a
theta3 = pi/2 - angle_b - atan2(0.054,1.5)
theta2_s =theta2.evalf(subs={q1: 1.5, q2: 2.5, q3: 3.0, q4: 3.14, q5: 2.14, q6: 1.14})
theta3_s =theta3.evalf(subs={q1: 1.5, q2: 2.5, q3: 3.0, q4: 3.14, q5: 2.14, q6: 1.14})

print("theta2 = ",theta2_s)
print("theta3 = ",theta3_s) 

R0_3 = T0_3[0:3,0:3]
R0_3 = R0_3.evalf(subs={q1: 1.5, q2: 2.5, q3: 3.0, q4: 3.14, q5: 2.14, q6: 1.14})
R3_6 = R0_3.inv("LU") * R0_6 
#T3_6 = T3_4 * T4_5 * T5_6
#R3_6 = T3_6[0:3,0:3]
print("R3_6 = ",R3_6.evalf(subs={q1: 0.5, q2: 0, q3: 0, q4: 0, q5: 0, q6: 0}))

theta4 = atan2(R3_6[2,2],-R3_6[0,2])
theta5 = atan2(sqrt(R3_6[0,2]**2+R3_6[2,2]**2),R3_6[1,2])
theta6 = atan2(-R3_6[1,1],R3_6[1,0])

theta4_s =theta4.evalf(subs={q1: 1.5, q2: 2.5, q3: 3.0, q4: 3.14, q5: 2.14, q6: 1.14})
theta5_s =theta5.evalf(subs={q1: 1.5, q2: 2.5, q3: 3.0, q4: 3.14, q5: 2.14, q6: 1.14})
theta6_s =theta6.evalf(subs={q1: 1.5, q2: 2.5, q3: 3.0, q4: 3.14, q5: 2.14, q6: 1.14})

print("theta4 = ",theta4_s)
print("theta5 = ",theta5_s)
print("theta6 = ",theta6_s)  




