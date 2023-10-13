# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 12:37:34 2023

@author: chmrj
"""

import matplotlib.pyplot as plt
import glob
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import seaborn as sns


#To convert raw deposition data into normalised absorbance, background substracted, with 3 gaussian fit

#define folder tructure

path = 'processed'
export_path='exports'

#   glob processed files
temperatures = glob.glob(path)
print(temperatures)

data_files = glob.glob('processed/***.txt')


#   Gaussian formula
def gaussian(x, A, x0, sig):
    return A*np.exp(-(x-x0)**2/(2*sig**2))

def multi_gaussian(x, *pars):
    g1 = gaussian(x, pars[0], pars[1], pars[2])
    g2 = gaussian(x, pars[3], pars[4], pars[5])
    g3 = gaussian(x, pars[6], pars[7], pars[8])
    g4 = gaussian(x, pars[9], pars[10], pars[11])
    # return g1 + g2 + g3 
    return g1 + g2 + g3 + g4

#   initial guess with bounds
init_vals = [0.2, 144,5,
         0.03, 155, 1,
         0.3, 120, 20,
         0.05, 180, 15]

bounds_min = [0.05,143,0,
              0,154,0,
              0,110,10,
              0,160,5]

bounds_max = [0.9,150,15,
              0.4,170,10,
              1.3,130,30,
              0.5,200,45]

#   create lists for data, temp & deposition
data_dict=[]
temp = []
depo = []


#   Load files and fit

for i in data_files:
    
    #   find temperature from data file name
    temperature = int(i[16:19])
    
    #   adjust for file naming - i.e. add  0.5 K to 127.5 and 137.5 K
    temperature = np.where((temperature==127 or temperature==137) , temperature+0.5, temperature)
    print(temperature)

    #   load data in datafram
    df = pd.read_csv(i, delimiter=' ', names=['wavelength', 'absorbance'])
    print(i)
   
    #   find data file name
    name = i[12:-4]

    #   find deposition number from file name
    d = i.find('dep')
    deposition = int(i[d+3])

    #   discard data below 117 nm and above 220 nm
    df = df.drop(df[df['wavelength']<117].index)


    #   set a linear background by finding the minimum point on the curve and drawing a linear line here
    linear_BkGd=df['absorbance'].min()
    #   substract bkg
    y = df['absorbance_bk'] = df['absorbance'] - linear_BkGd  
    # #make shorthand for wavelength
    x= df['wavelength']

    #   fit Gaussian
    popt, pcov = curve_fit(multi_gaussian, x, df['absorbance_bk'], p0=init_vals, bounds=(bounds_min,bounds_max))
    
    perr_3gauss = np.sqrt(np.diag(pcov))

    pars_1 = popt[0:3]
    pars_2 = popt[3:6]
    pars_3 = popt[6:9]
    pars_4 = popt[9:12]
    gauss_peak_1 = gaussian(x, *pars_1)
    gauss_peak_2 = gaussian(x, *pars_2)
    gauss_peak_3 = gaussian(x, *pars_3)
    gauss_peak_4 = gaussian(x, *pars_4)
    
    # #calculate errors and residuals:
    df['residual_3gauss']=residual_2gauss = y - (multi_gaussian(x, *popt))

    residuals=sum((df['absorbance_bk']-(multi_gaussian(df['wavelength'], *popt)))**2)

    #   fit parameters
    #---------------------
    pars_err_1 = perr_3gauss[0:3] # divide the output parameteres into to arrays, one for the first Gaussian; a 2nd for Gaussian 2; etc
    pars_err_2 = perr_3gauss[3:6]
    pars_err_3 = perr_3gauss[6:9]
    pars_err_4 = perr_3gauss[9:12]
    
    #   Concatenate fits
    Gauss = np.concatenate((pars_1, pars_err_1, pars_2, pars_err_2, pars_3, pars_err_3, pars_4, pars_err_4))

    #   Append Gaus, Temp and Dep to respective lists
    data_dict.append(Gauss)
    temp.append(temperature)
    depo.append(deposition)


    # plot each fit, file by file
    fig, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios':[3,1]}) # I am making a 2 x 1 row x column grid, when the first row is 3 times the height ofthe2nd
    fig.subplots_adjust(hspace=0) # Remove horizontal space between axes

    fig.suptitle('fit data with 3 Gaussians'+name, family="serif", fontsize=12)
    plt.xlabel('wavelength / nm', family="serif", fontsize=12)

    axs[1].plot(x,df['residual_3gauss'],'go:',label='res')

    axs[1].set_ylabel("residuals",family="serif", fontsize=12)  
    axs[0].plot(x,df['absorbance_bk'],'b+:',label='data')

    axs[0].plot(x, gauss_peak_1, "g")
    axs[0].fill_between(x, gauss_peak_1.min(), gauss_peak_1, facecolor="green", alpha=0.5)

    axs[0].set_ylabel("absorbance",family="serif", fontsize=12)    
    axs[0].plot(x,  gauss_peak_2, "y")
    axs[0].fill_between(x, gauss_peak_2.min(), gauss_peak_2, facecolor="yellow", alpha=0.5)  

    axs[0].plot(x,  gauss_peak_3, "m")
    axs[0].fill_between(x, gauss_peak_3.min(), gauss_peak_3, facecolor="magenta", alpha=0.5)  

    axs[0].plot(x,  gauss_peak_4, "k")
    axs[0].fill_between(x, gauss_peak_4.min(), gauss_peak_4, facecolor="k", alpha=0.5)  

    axs[0].annotate(residuals, xy =(150, max(df['absorbance_bk'])))

    axs[1].legend()
    axs[0].legend()


    plt.show() #This command would plot a different graph for each f during the cycling

    # Save the dataframe to csv file
    df.to_csv('../exports/'+name+'.csv', index=False)


data_dict = np.array(data_dict)
temp = np.array(temp)
depo = np.array(depo)
data = np.round(np.concatenate((temp[:,None], depo[:,None],data_dict), axis=1),8)

data_dict=dict(enumerate(data,0))


#    plot gaussian peak max as a function of temperatue
#--------------------------------------------------------

fig, axs = plt.subplots(4, 1, sharex=True, figsize=(4,7)) # I am making a 2 x 1 row x column grid
fig.subplots_adjust(hspace=0.1)
plt.xlabel('temperature / K', family="garamond", fontsize=18)

#ax.errorbar(x, z, markersize=10, xerr=sigx, yerr=sigy, fmt='.', color='red', label='observed')
# axs[0].errorbar(all_temp, Gauss_top_pos, yerr=None, xerr = None , fmt='bo', label='Peak 1')

p = pd.DataFrame(data, columns=['temp','dep',
                                'ab1','peak1','sig1','ab1_er','peak1_er','sig1_er',
                                'ab2','peak2','sig2','ab2_er','peak2_er','sig2_er',
                                'ab3','peak3','sig3','ab3_er','peak3_er','sig3_er',
                                'ab4','peak4','sig4','ab4_er','peak4_er','sig4_er'])

plt.rcParams['errorbar.capsize']=2
#   115 nm peak
p.loc[(p.ab1>0.47) & (p.ab1<0.71)].plot.scatter(x='temp',y='peak1', yerr = 'peak1_er', ax=axs[1])
#   144 nm peak
p.loc[(p.ab1>0.47) & (p.ab1<0.71)].plot.scatter(x='temp',y='peak2', yerr = 'peak2_er',ax=axs[2])
#   155 nm peak
p.loc[(p.ab1>0.47) & (p.ab1<0.71)].plot.scatter(x='temp',y='peak3', yerr = 'peak3_er',ax=axs[0])
#   190 nm peak
p.loc[(p.ab1>0.47) & (p.ab1<0.71)].plot.scatter(x='temp',y='peak4', yerr = 'peak4_er',ax=axs[3])

plt.show()

# # #save to text file
p.to_csv('gaussian_fits.csv', index=False)


#   plot spont and nonspont Gauss peak as function of absorbance
#----------------------------------------------------------------------
fig, axs = plt.subplots(2, 1, sharex=False, figsize=(4,6)) # I am making a 2 x 1 row x column grid
fig.subplots_adjust(hspace=0.4)

#   144 nm peak
sns.lineplot(data=p.loc[p.temp>=125], x='ab1',y='peak1', marker='o', hue='temp', ax=axs[0])
#   155 nm peak
sns.lineplot(data=p.loc[p.temp<125], x='ab2',y='peak2', marker='o', hue='temp', ax=axs[1])

axs[0].set_xlabel('Absorbance')
axs[0].set_title('Spont temps (>= 125 K)')
axs[1].set_title('Non-spont temps (<125 K)')
axs[1].set_xlabel('Absorbance')

plt.show()

