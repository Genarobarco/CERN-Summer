#%%

import os
import sys
import shutil
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math as math
from glob import glob
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.special import factorial

def lineal(x, a, b):
    return a*x+b

ruta = r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\Current_ArN2_9010_5bar.xlsx"

data = pd.read_excel(ruta, header = 0,
                     names = ['Voltage', 'Current', 'Segunda', 'Difference'])

plt.figure(figsize = (12,8))
plt.scatter(data['Voltage'], data['Current'], 
            color = 'crimson', label='First')
plt.scatter(data['Voltage'], data['Segunda'], 
            color = 'navy', label='Second')
plt.xlabel('Voltage (V)', fontsize=15)
plt.ylabel('Current (A)',fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.grid()
plt.legend(fontsize=15)

plt.savefig(f'ArN2_9010_5bar_CV.jpg', format='jpg', 
            bbox_inches='tight', dpi = 300)

plt.show()

plt.figure(figsize = (12,8))
plt.scatter(data['Voltage'], data['Difference'], 
            color = 'green', label='Difference')
plt.xlabel('Voltage (V)', fontsize=15)
plt.ylabel('Current (A)',fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.grid()
plt.legend(fontsize=15)

plt.savefig(f'ArN2_9010_5bar_Diferencia.jpg', format='jpg', 
            bbox_inches='tight', dpi = 300)

plt.show()