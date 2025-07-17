#%%

import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math as math
from glob import glob

def Excel_value(file_path, filters, target_column):

    df = pd.read_excel(file_path)
    
    mask = pd.Series(True, index=df.index)
    for col, val in filters.items():
        mask &= (df[col] == val)
    
    filtered_df = df[mask]
    
    if filtered_df.empty:
        print("No match found with the given filters.")
        return None
    
    return filtered_df[target_column].values[0]  # Return the first match

def RP(base_path):
  results_path = os.path.join(base_path, 'Results')
  matched_files = glob(os.path.join(results_path, '*-calibratedResults.csv'))

  if matched_files:
      
      df_results = pd.read_csv(matched_files[0], sep = ',')

      df = pd.DataFrame({
        'Lambda': df_results['wavelength'],
        'Counts': df_results['intensity'],
        'Counts_norm': df_results['intensity']/max(df_results['intensity'])
        })

      return df
  else:
      return None

excel_path = r'C:\Users\genar\VSC code\CERN-Summer\Whole_Data.xlsx'

filters = {
    'Element A': 'Ar',
    'Concentration A': 100,
    'Element B': 'Ar',
    'Concentration B': 0,
    'Pressure (bar)': 5.0
}

Saturation_current =Excel_value(excel_path, filters, 'SC')

N_e = Saturation_current / (-1.602176634e-19)

R_1 = r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\Ar\100\5_bar\40kV40mA\0V"
R_2 = r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\Ar\100\5_bar\40kV40mA\0V_try2"
R_3 = r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\Ar\100\5_bar\40kV40mA\0V_try3"
R_4 = r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\Ar\100\5_bar\40kV40mA\0V_try4"

df_1= RP(R_1)
df_2= RP(R_2)
df_3= RP(R_3)
df_4= RP(R_4)

resta_2 = (df_2['Counts'] - df_1['Counts'])/N_e
resta_3 = (df_3['Counts'] - df_1['Counts'])/N_e
resta_4 = (df_4['Counts'] - df_1['Counts'])/N_e

plt.figure(figsize=(12,8))

plt.plot(df_2['Lambda'], resta_2, 
         label='BG/S pairs', color='blue', linewidth = 0.5)

plt.plot(df_3['Lambda'], resta_3, 
         label='S - wait - BG', color='green', linewidth = 0.5)

plt.plot(df_4['Lambda'], resta_4, 
         label='S/BG', color='magenta', linewidth = 0.5)

plt.legend(fontsize=20)
plt.xlabel('Wavelenght', fontsize=15)
plt.ylabel(r'Photons per e$^-$ (A.U.)', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.grid()

# plt.savefig(f'WithoutFirstContinuom.jpg', format='jpg', 
#             bbox_inches='tight', dpi = 300) 

plt.show(block=False)
plt.pause(0.1)
input("Press enter to close all figures...")
plt.close('all')