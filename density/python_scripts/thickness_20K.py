# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.ticker as ticker
import matplotlib

matplotlib.rcParams.update({'font.size': 12, 'font.family': 'serif'})

data = np.loadtxt('../20220531/T69824.dat', skiprows=2)

# print(data[:,0])

cf= 1.

time = data[80:400,0]
# print(time)

def pies (x,*p):
    y0 = p[0]
    xc = p[1]
    w  = p[2]
    A =p  = p[3]
    return y0+A*np.sin(np.pi*(x-xc)/w)




# print(int[23,0]*1.5)

# t_s = 30
# t_f = 270



# plt.plot(data[t_s-10:t_f+10,0], data[t_s-10:t_f+10,3])



guess = [10,1,100,10]

popt, pcov = curve_fit(pies, time, data[80:400,3], p0=guess, maxfev=20000)
y0, xc, w, A = popt

y=y0+A*np.sin(2*np.pi*(data[80:400,0]-xc)/w)

# # ax = plt.axes()
# # formatter = ticker.ScalarFormatter(useMathText=True)
# # formatter.set_scientific(True)
# # formatter.set_powerlimits((-1,1))
# # ax.yaxis.set_major_formatter(formatter)
# # ax.xaxis.set_major_locator(ticker.MultipleLocator(40))
# # ax.xaxis.set_minor_locator(ticker.MultipleLocator(20))
# # ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
# # ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
# # ax.yaxis.set_ticks_position('both')
# # ax.xaxis.set_ticks_position('both')
# # ax.tick_params(which='both', direction='in')

# plt.grid(True, which='both', axis='both', color='k', linestyle='--', linewidth=0.1)


plt.plot(time, data[80:400,3], '.', label='data' )
plt.plot(time, pies(time,*popt),'r-', label='Fit')
plt.xlabel('Time (s)')
plt.ylabel(r'Diode Voltage (mV)')
plt.show()


print('guess =', guess)

chi_squared = np.sum(((data[80:400,3]-y)**2)/y)
print ("chi-squared =",chi_squared)

print("\ny0 =", popt[0], "\nxc =", popt[1], "\nw =", popt[2], "\nA =", popt[3])

n2=(y0+A)/(y0-A)

print ("\nRefractive index =",n2, "at 632.8 nm")

n2=1.1985

theta=np.radians(20)

theta2=np.arcsin(theta/n2)
theta2_deg=np.rad2deg(theta2)

print ("\ntheta2 =", theta2, "rad \ntheta2 =", theta2_deg, "degrees")

d=632.8/(2*n2*np.cos(theta2))
print ("\nd =", d, "nm")

rate=d/(2*w)

print ("\nDeposition rate =", rate, "nm/s")

# t= (t_f-t_s)/1.5
t=15

thickness=rate*t

print ("\nThickness =", thickness, "nm\n\nNow get out of the lab and have some fun!\n")