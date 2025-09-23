# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 15:28:52 2022

@author: shd-isa-uv
"""

import numpy as np
import matplotlib.pyplot as plt


x = 20,30,40,50,70,80,90,100,110, 120,125
z = 0.582, 0.602, 0.745, 0.658, 0.794, 0.85, .821, .864, .961, .953, .908




plt.ylabel ('Density g cm-3')
plt.xlabel('Td')


y = 17.78, 15.361,13.872, 15.716, 13.01, 12.16, 12.59, 11.97, 10.758, 10.85, 11.391
y =np.array(y)

y = (11/y)*0.94
# plt.scatter(x,y)

print(y)

x5 = [20,
      30,
      40,
      50,
      70,
      80,
      90,
      100,
      110,
      120,
      125]
a_list = [144.277,
      144.099,
      143.962,
      143.853,
      143.72,
      143.6,
      143.528,
      143.540,
      143.54,
      143.646,
      143.266,
      143.659,
      ]

# y5 = np.array()

# plt.scatter(1/1e7/(y5-144.277), y)

y  = 0, 85.6174, 151.658, 204.291, 268.622, 326.766, 361.7, 355.875, 355.875,304.466, 489.115

# y = np.array(y)

plt.scatter(x5,z)


# a = np.c_[y,z]
# 
# print(a)


# 298.166 

# new_list = []
# for i, num in enumerate(a_list[:-1]):  
#     new_num = num - a_list[i+1]
#     new_list.append(new_num)
    
# print(new_list)