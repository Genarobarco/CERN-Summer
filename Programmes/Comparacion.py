#%%

import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math as math
from glob import glob

Ruta_1 = r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\CF4\1\5_bar\40kV40mA\Gena_measurements\try1\Results\Candela-0.5781uA-calibratedResults.csv"
Ruta_2 = r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\CF4\1\5_bar\40kV40mA\Gena_measurements\try2\Results\try2-0.72uA-calibratedResults.csv"
Ruta_3 = r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\CF4\1\5_bar\40kV40mA\Gena_measurements\try3\Results\try3-0.72uA-calibratedResults.csv"
Ruta_4 = r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\CF4\1\5_bar\40kV40mA\Gena_measurements\try4\Results\try4-0.72uA-calibratedResults.csv"

df_1 = pd.read_csv(Ruta_1, sep = ',')

wavelength_1 = df_1['wavelength']
intensity_1 = df_1['intensity']

df_2 = pd.read_csv(Ruta_2)

wavelength_2 = df_2['wavelength']
intensity_2 = df_2['intensity']

df_3 = pd.read_csv(Ruta_3, sep = ',')

wavelength_3 = df_3['wavelength']
intensity_3 = df_3['intensity']

df_4 = pd.read_csv(Ruta_4, sep = ',')

wavelength_4 = df_4['wavelength']
intensity_4 = df_4['intensity']

intensity_norm_1 = intensity_1 / max(intensity_1)
intensity_norm_2 = intensity_2 / max(intensity_2)
intensity_norm_3 = intensity_3 / max(intensity_3)
intensity_norm_4 = intensity_4 / max(intensity_4)

plt.figure(figsize=(12,8))

# plt.plot(wavelength_1, intensity_1, 
#          label=f'Yesterday', color='red', linewidth = 0.5)

plt.plot(wavelength_2, intensity_norm_2, 
         label=rf'6.064v', color='navy', linewidth = 0.5)

plt.plot(wavelength_3, intensity_norm_3, 
         label=rf'6v', color='green', linewidth = 0.5)

plt.plot(wavelength_4, intensity_norm_4, 
         label=rf'6v - again', color='orange', linewidth = 0.5)

plt.legend(fontsize=20, loc = 'lower center')
plt.xlabel('Wavelenght', fontsize=15)
plt.ylabel('Absolute Counts (A.U.)', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)

plt.grid()

# plt.savefig('Detail_WaterPik_ArN2_Absolute.jpg', format='jpg',
#             bbox_inches = 'tight', dpi = 300)

plt.show(block=False)
plt.pause(0.1)
input('Press enter to close all figures...')

plt.close('all')
