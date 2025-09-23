# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 17:20:26 2022

@author: shd-isa-uv
"""

import matplotlib.pyplot as plt
import numpy as np

x = 20, 70, 90, 120, 30, 40, 50, 125, 80, 100, 110, 127.5, 60
y = 17.78, 13.01, 12.59, 10.85, 15.361, 13.872, 15.713, 11.391, 12.160, 11.97, 10.758, 10.08, 13.45

y = np.array(y)
plt.scatter(x, y)

plt.xlabel('Td')
plt.ylabel('Film thickness for first deposition (nm)')
