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

def Mean_BG(Rutas):

  Count_List=[]
  Lambda_List=[]
  Indice= 0

  for ruta in Rutas:

    array= pd.read_csv(ruta, names=['Lambda', 'Counts'], sep='\s+', 
                      header=None, engine='python')
    
    Count_List.append(array['Counts'])
    Lambda_List.append(array['Lambda'])

    Indice += 1

  mean_per_channel=[]
  std_per_channel=[]

  # Convertimos la lista de cuentas en un array 2D (archivos x canales)
  counts_matrix = np.array(Count_List)

  # Calculamos media y desviación por canal (eje 0 = entre archivos)
  mean_per_channel = np.mean(counts_matrix, axis=0)
  std_per_channel = np.std(counts_matrix, axis=0)

  channels=Lambda_List[0]

  df = pd.DataFrame({
    'Lambda': channels,
    'Mean_Counts': mean_per_channel,
    'Std_Counts': std_per_channel
  })

  return df

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
            'Counts_norm': df_rute['intensity']/max(df_rute['intensity'])
            })
        data[i] = df
    else:
        print(f'{i} file did not found')

  return data

Ruta_cal = r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\Calibration_old\Calibration.txt"
Ruta = r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\CYGNO\HeCF4-60-40\100_mbar_5"
Ruta_BG=glob(f'{Ruta}\DataBG\*.txt')

Calibration = pd.read_csv(Ruta_cal, sep = '\t', header= None,
                          names=['Lambda', 'Counts'])

Cal_deriv = pd.read_csv(r'C:\Users\genar\VSC code\CERN-Summer\Programmes\Cal_derivated.txt', 
                        sep = '\t', header= None, names=['Lambda', 'Counts'])

filtered_data = RP(Ruta)
df_MeanBG = Mean_BG(Ruta_BG)

df_results = filtered_data['calibratedResults']
df_bgSpec = filtered_data['bgSpectrum']             # raw Background
df_corrSpec = filtered_data['correctedSpectrum']
df_raw = filtered_data['rawSpectrum']               # raw Signal

diff_BGs = df_bgSpec['Counts']-df_MeanBG['Mean_Counts']

#%%

print(len(df_results['Lambda']))
print(len(df_bgSpec['Lambda'][57:1810]))

#%%

# plt.figure(figsize=(12,8))

# plt.plot(Calibration['Lambda'], Cal_deriv, 
#         label=f'Prueba', color='crimson', linewidth = 0.5, alpha = 1)

# plt.legend(fontsize=20)
# plt.xlabel('Wavelenght', fontsize=15)
# plt.ylabel('Absolute Counts (A.U.)', fontsize=15)
# plt.tick_params(axis='both', which='major', labelsize=15)
# plt.grid()