#%%

import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math as math
from glob import glob

Ruta_1 = r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\N2\100\1_bar\40kV40mA\Non_Alternate\Results\Non_Alternate-1uA-calibratedResults.csv"
Ruta_2 = r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\N2\100\1_bar\40kV40mA\Alternate\Results\Alternate-1uA-calibratedResults.csv"


df_1 = pd.read_csv(Ruta_1)

wavelength_NoAlt = df_1['wavelength']
intensity_NoAlt = df_1['intensity']

intensity_norm_NoAlt = intensity_NoAlt

df_2 = pd.read_csv(Ruta_2)

wavelength_Alt = df_2['wavelength']
intensity_Alt = df_2['intensity']

intensity_norm_Alt = intensity_Alt

plt.figure(figsize=(12,8))

plt.plot(wavelength_NoAlt, intensity_norm_NoAlt / max(intensity_norm_NoAlt) , 
         label=f'No Alt', color='green')

plt.plot(wavelength_Alt, intensity_norm_Alt / max(intensity_norm_Alt) , 
         label=f'Alternado', color='blue')

plt.legend(fontsize=20)
plt.xlabel('Wavelenght', fontsize=15)
plt.ylabel('Normalize Counts (A.U.)', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)

plt.show()
