#%%

import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math as math
from glob import glob

def RP(base_path):
  
  results_path = os.path.join(base_path, 'Results')
  files = ['calibratedResults','bgSpectrum', 'correctedSpectrum', 'rawSpectrum']

  data = {}

  for i in files:

    matched_files = glob(os.path.join(results_path, f'*-{i}.csv'))

    if matched_files:
        df_rute = pd.read_csv(matched_files[0], sep = ',')
        df = pd.DataFrame({
            'Lambda': df_rute['wavelength'],
            'Counts': df_rute['intensity'],
            'Counts_norm': df_rute['intensity']/max(df_rute['intensity']),
            'Err_Counts': np.sqrt(df_rute['intensity'].clip(lower=0)),
            })
        data[i] = df
    else:
        print(f'{i} file did not found')

  return data

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
R_3 = r'C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\N2\100\1_bar\40kV40mA\0V_Again'

df_1=RP(R_1)['calibratedResults']
df_2=RP(R_2)['calibratedResults']
df_3=RP(R_3)['calibratedResults']
df_4=RP(R_3)['calibratedResults']

plt.figure(figsize=(12,8))

plt.plot(df_1['Lambda'], df_1['Counts'].clip(lower = 0), 
         label=rf'7/7', color='blue', linewidth = 0.5)

plt.plot(df_2['Lambda'], df_2['Counts'].clip(lower = 0), 
         label=rf'14/7', color='red', linewidth = 0.5)

plt.plot(df_3['Lambda'], df_3['Counts'].clip(lower = 0), 
         label=rf'After WC', color='green', linewidth = 0.5)

plt.plot(df_4['Lambda'], df_4['Counts'].clip(lower = 0), 
         label=rf'Current list', color='darkviolet', linewidth = 0.5)

plt.legend(fontsize=20)
plt.xlabel('Wavelenght', fontsize=15)
plt.ylabel('Absolute Counts (A.U.)', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)

plt.grid()

# plt.savefig('N2_100_1bar_Comp_B&A_windowchange.jpg', format='jpg',
#             bbox_inches = 'tight', dpi = 300)

plt.show(block=False)
plt.pause(0.1)
input('Press enter to close all figures...')

plt.close('all')
