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


data_name=['Data0']

product_list=[]

for i in data_name:

    Data_file = rf'{Data_folder}\Data\{i}.txt'
    BG_file = rf'{Data_folder}\DataBG\{i}.txt'

    dt_cal = pd.read_csv(Calibration_file, sep='\t', header = None, names = ['w', 'i'])
    dt_data = pd.read_csv(Data_file, sep='\t', header = None, names = ['w', 'i'])
    dt_BG = pd.read_csv(BG_file, sep='\t', header = None, names = ['w', 'i'])

    wavelength = dt_cal['w']

    cal_coeficient = dt_cal['i'] 
    counts_data = dt_data['i']
    counts_bg = dt_BG['i']

    resta = counts_data - counts_bg

    product = resta * cal_coeficient

product_list.append(product)

product_matrix = np.array(product_list)

# Calculamos media y desviación por canal (eje 0 = entre archivos)
mean_per_channel = np.mean(product_matrix, axis=0)

plt.plot(wavelength, mean_per_channel)