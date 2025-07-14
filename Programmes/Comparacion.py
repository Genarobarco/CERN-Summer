#%%

import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math as math
from glob import glob

Ruta_1 = r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\Ar\100\5_bar\40kV40mA\0V\Results\0V-0.63uA-calibratedResults.csv"
Ruta_2 = r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\N2\0\1bar\40kV40mA\Results\40kv40mA-1uA-calibratedResults.csv"
Ruta_3 = r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\CF4\0\1_bar\40kV40mA\0_V\run_alternate\run1\Results\run1-1uA-calibratedResults.csv"


df_1 = pd.read_csv(Ruta_1, sep = ',')

wavelength_1 = df_1['wavelength']
intensity_1 = df_1['intensity'].clip(lower=0)
error_1 = np.sqrt(intensity_1) 

df_2 = pd.read_csv(Ruta_2)

wavelength_2 = df_2['wavelength']
intensity_2 = df_2['intensity']

df_3 = pd.read_csv(Ruta_3, sep = ',')

wavelength_3 = df_3['wavelength']
intensity_3 = df_3['intensity']

# df_4 = pd.read_csv(Ruta_4, sep = ',')

# wavelength_4 = df_4['wavelength']
# intensity_4 = df_4['intensity']

# df_5 = pd.read_csv(Ruta_5, sep = ',')

# wavelength_5 = df_5['wavelength']
# intensity_5 = df_5['intensity']

# df_6= pd.read_csv(Ruta_6, sep = ',')

# wavelength_6 = df_6['wavelength']
# intensity_6 = df_6['intensity']

intensity_norm_1 = intensity_1 / max(intensity_1)
intensity_norm_2 = intensity_2 / max(intensity_2)
intensity_norm_3 = intensity_3 / max(intensity_3)
# intensity_norm_4 = intensity_4 / max(intensity_4)
# intensity_norm_5 = intensity_5 / max(intensity_5)
# intensity_norm_6 = intensity_6 / max(intensity_6)

plt.figure(figsize=(12,8))

plt.plot(wavelength_1, intensity_1, 
         label=f'12/7 - 5bar', color='blue', linewidth = 0.5)

# plt.plot(wavelength_2, intensity_2, 
#          label=rf'8/7 - 1bar - N2', color='blue', linewidth = 0.5)

# plt.plot(wavelength_3, intensity_3, 
#          label=rf'30/6 - 1bar - CF4', color='green', linewidth = 0.5)

# plt.plot(wavelength_4, intensity_4, 
#          label=rf'S/BG pairs - again', color='magenta', linewidth = 0.5)

# plt.plot(wavelength_5, intensity_5, 
#          label=rf'try5 - 6v - pairs S/BG', color='magenta', linewidth = 0.5)

# plt.plot(wavelength_6, intensity_6, 
#          label=rf'try6 - waiting 1 file', color='crimson', linewidth = 0.5)

plt.legend(fontsize=20, loc = 'upper center')
plt.xlabel('Wavelenght', fontsize=15)
plt.ylabel('Absolute Counts (A.U.)', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)

plt.grid()

# plt.savefig('ArPure_measurements_duringtime.jpg', format='jpg',
#             bbox_inches = 'tight', dpi = 300)

plt.show(block=False)
plt.pause(0.1)
input('Press enter to close all figures...')

plt.close('all')
