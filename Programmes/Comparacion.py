#%%

import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math as math
from glob import glob

Ruta_1 = r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\CF4\1\5_bar\40kV40mA\Candela\Results\Candela-0.5781uA-calibratedResults.csv"
Ruta_2 = r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\N2\0\1bar\40kV40mA\Results\40kv40mA-1uA-calibratedResults.csv"

df_1 = pd.read_csv(Ruta_1, sep = ',')

wavelength_1 = df_1['wavelength']
intensity_1 = df_1['intensity']

df_2 = pd.read_csv(Ruta_2)

wavelength_2 = df_2['wavelength']
intensity_2 = df_2['intensity']

intensity_norm_1 = intensity_1 / max(intensity_1)
intensity_norm_2 = intensity_2 / max(intensity_2)

plt.figure(figsize=(12,8))

plt.plot(wavelength_1, intensity_1, 
         label=f'Ar/CF4 - 99/1 - 5bar', color='red', linewidth = 0.5)

plt.plot(wavelength_2, intensity_2, 
         label=rf'Ar pure - yesterday', color='navy', linewidth = 0.5)

plt.legend(fontsize=20, loc = 'upper left')
plt.xlabel('Wavelenght', fontsize=15)
plt.ylabel('Absolute Counts (A.U.)', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)

plt.grid()

# plt.savefig('Detail_WaterPik_ArN2_Absolute.jpg', format='jpg',
#             bbox_inches = 'tight', dpi = 300)

plt.show()
