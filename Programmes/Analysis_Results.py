#%%
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import DataSorting as DS

Ruta=r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\N2\5\1bar\40kV40mA"

Ruta_Results = DS.RP(Ruta)

name = 'N2_955_1b'

df_1 = pd.read_csv(Ruta_Results)

wavelength = df_1['wavelength']
intensity = df_1['intensity']

intensity_norm = intensity / max(intensity)
#%%


intensity.sort_values()

plt.plot(wavelength, intensity.sort_values())


#%%

plt.figure(figsize=(12,8))

plt.plot(wavelength, intensity_norm , 
         label=f'{name}', color='navy', linewidth = 0.5, alpha = 1)
plt.legend(fontsize=20)
plt.xlabel('Wavelenght', fontsize=15)
plt.ylabel('Normalize Counts (A.U.)', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.grid()

plt.show()