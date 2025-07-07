#%%
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math as math
from glob import glob


Data_folder = r'C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\N2\100\1_bar\40kV40mA\Alternate'

Calibration_file = r'C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\Calibration_old\Calibration.txt'


Data_fs = glob(rf'{Data_folder}\Data\*.txt')
BG_fs = glob(rf'{Data_folder}\DataBG\*.txt')

confirm = False

if len(Data_fs)==len(BG_fs):
    confirm = True
else:
    print('Data and DataBG have different amount of files.')



dt_cal = pd.read_csv(Calibration_file, sep='\t', header = None, names = ['w', 'i'])
dt_data = pd.read_csv(Data_file, sep='\t', header = None, names = ['w', 'i'])
dt_BG = pd.read_csv(BG_file, sep='\t', header = None, names = ['w', 'i'])

wavelength = dt_cal['w']

cal_coeficient = dt_cal['i'] 
counts_data = dt_data['i']
counts_bg = dt_BG['i']

resta = counts_data - counts_bg

product = resta * cal_coeficient

plt.plot(wavelength, product)