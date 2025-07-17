#%%

import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math as math
from glob import glob

def RP(base_path):

  """
  Returns the full path to the first file in the 'Results' subfolder 
  of `base_path` that ends with '-calibratedResults.csv'.

  Parameters:
      base_path (str): Path to the folder containing the 'Results' subfolder.

  Returns:
      str: Full path to the matching file, or None if not found.
  """
  results_path = os.path.join(base_path, 'Results')
  matched_files = glob(os.path.join(results_path, '*-calibratedResults.csv'))

  if matched_files:
      return matched_files[0]
  else:
      return None

def data_pablo(Concentration, Pressure, Tube_intensity, Voltage):

    path_datos = r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\Dataframe_Spectra_ArCF4"
    datos = pd.read_pickle(path_datos)

    mask1 = datos['Concentrations'] == Concentration 
    mask2 = datos['Pressure'] == Pressure
    mask3 = datos['Tube Intensity'] == Tube_intensity
    mask4 = datos['Voltages'] == Voltage
    fila = datos[mask1 & mask2 & mask3 & mask4]['Data']

    wave_pablo = fila.array[0][:,0]
    inte_pablo = fila.array[0][:,1]

    inte_pablo_norm = inte_pablo / max(inte_pablo)

    df_Pablo = pd.DataFrame({
        'Lambda': wave_pablo,
        'Counts': inte_pablo,
        'Counts_norm': inte_pablo_norm
        })
    
    return df_Pablo

df_pablo = data_pablo('Ar_95_CF4_5', 5.0, '40kV40mA', 0)

R_1 = r'C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\N2\100\1_bar\40kV40mA\Alternate'
R_2 = r'C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\N2\100\1_bar\40kV40mA\0V'
R_3 = r'C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\N2\100\1_bar\40kV40mA\New_Window'

Ruta_1=RP(R_1)
Ruta_2=RP(R_2)
Ruta_3=RP(R_3)


df_1 = pd.read_csv(Ruta_1, sep = ',')

wavelength_1 = df_1['wavelength']
intensity_1 = df_1['intensity'].clip(lower=0)
error_1 = np.sqrt(intensity_1) 

df_2 = pd.read_csv(Ruta_2)

wavelength_2 = df_2['wavelength']
intensity_2 = df_2['intensity'].clip(lower=0)

df_3 = pd.read_csv(Ruta_3, sep = ',')

wavelength_3 = df_3['wavelength']
intensity_3 = df_3['intensity'].clip(lower=0)

# df_4 = pd.read_csv(Ruta_4, sep = ',')

# wavelength_4 = df_4['wavelength']
# intensity_4 = df_4['intensity'].clip(lower=0)

# df_5 = pd.read_csv(Ruta_5, sep = ',')

# wavelength_5 = df_5['wavelength']
# intensity_5 = df_5['intensity'].clip(lower=0)

# df_6 = pd.read_csv(Ruta_6, sep = ',')

# wavelength_6 = df_6['wavelength']
# intensity_6 = df_6['intensity'].clip(lower=0)

# df_7 = pd.read_csv(Ruta_7, sep = ',')

# wavelength_7 = df_7['wavelength']
# intensity_7 = df_7['intensity'].clip(lower=0)

# df_8 = pd.read_csv(Ruta_8, sep = ',')

# wavelength_8 = df_8['wavelength']
# intensity_8 = df_8['intensity'].clip(lower=0)

intensity_norm_1 = intensity_1 / max(intensity_1)
intensity_norm_2 = intensity_2 / max(intensity_2)
intensity_norm_3 = intensity_3 / max(intensity_3)
# intensity_norm_4 = intensity_4 / max(intensity_4)
# intensity_norm_5 = intensity_5 / max(intensity_5)
# intensity_norm_6 = intensity_6 / max(intensity_6)
# intensity_norm_7 = intensity_7 / max(intensity_7)
# intensity_norm_8 = intensity_8 / max(intensity_8)

plt.figure(figsize=(12,8))

# plt.plot(df_pablo['Lambda'], df_pablo['Counts'], 
#          label='Pablo', color='magenta', linewidth = 0.5)

plt.plot(wavelength_1, intensity_1, 
         label=rf'7/7', color='blue', linewidth = 0.5)

plt.plot(wavelength_2, intensity_2, 
         label=rf'14/7', color='red', linewidth = 0.5)

plt.plot(wavelength_3, intensity_3, 
         label=rf'After WC', color='green', linewidth = 0.5)

# plt.plot(wavelength_4, intensity_4, 
#          label=rf'same', color='violet', linewidth = 0.5)

# plt.plot(wavelength_5, intensity_5, 
#          label=rf'6v-50mbar', color='orange', linewidth = 0.5)

# plt.plot(wavelength_6, intensity_6, 
#          label=rf'same', color='crimson', linewidth = 0.5)

# plt.plot(wavelength_7, intensity_7, 
#          label=rf'same S/BG', color='navy', linewidth = 0.5)

# plt.plot(wavelength_8, intensity_8, 
#          label=rf'same BG/S', color='aqua', linewidth = 0.5)

plt.legend(fontsize=20)
plt.xlabel('Wavelenght', fontsize=15)
plt.ylabel('Absolute Counts (A.U.)', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)

plt.grid()

plt.savefig('N2_100_1bar_Comp_B&A_windowchange.jpg', format='jpg',
            bbox_inches = 'tight', dpi = 300)

plt.show(block=False)
plt.pause(0.1)
input('Press enter to close all figures...')

plt.close('all')
