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

excel_path = r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\Whole_Data.xlsx"

filters = {
    'Element A': 'Ar',
    'Concentration A': 100,
    'Element B': 'Ar',
    'Concentration B': 0,
    'Pressure (bar)': 5.0
}

Saturation_current = Excel_value(excel_path, filters, 'SC')
err_SC = 0.1e-7

#%%

R_1 = r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\Ar\100\5_bar\40kV40mA\0V"
R_2 = r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\Ar\100\5_bar\40kV40mA\0V_try2"
R_3 = r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\Ar\100\5_bar\40kV40mA\0V_try3"
R_4 = r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\Ar\100\5_bar\40kV40mA\0V_try4"

df_1= RP(R_1)['calibratedResults']
df_2= RP(R_2)['calibratedResults']
df_3= RP(R_3)['calibratedResults']
df_4= RP(R_4)['calibratedResults']

N_e = Saturation_current / (-1.602176634e-19)
err_NumeroElectrones = err_SC/ (-1.602176634e-19)

df_phe_1 = df_1['Counts'].clip(lower = 0)/N_e
err_phe_1 = np.sqrt((df_1['Err_Counts']/N_e)**2+(df_1['Counts']*err_NumeroElectrones/(N_e)**2)**2)

df_phe_2 = df_2['Counts'].clip(lower = 0)/N_e
err_phe_2 = np.sqrt((df_2['Err_Counts']/N_e)**2+(df_2['Counts']*err_NumeroElectrones/(N_e)**2)**2)

err_resta_phe = np.sqrt(err_phe_1**2+err_phe_2**2)

resta_2 = (df_2['Counts'] - df_1['Counts'])/N_e
resta_3 = (df_3['Counts'] - df_1['Counts'])/N_e
resta_4 = (df_4['Counts'] - df_1['Counts'])/N_e

resta_2 = resta_2.clip(lower=0)

df = pd.DataFrame({
    'Wavelength': df_1['Lambda'],
    '1': resta_2,
    '2': resta_3,
    '3': resta_4,
    'Error': err_phe_1
})

# Write the DataFrame to a .txt file (tab-separated)
df.to_csv('ArPure_WithOutContinuous.txt', sep='\t', index=False)


fig, axs = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={'width_ratios': [1, 1]})

# --- PLOT IZQUIERDO: df_1 y df_2 ---
ax_left = axs[0]

ax_left.plot(df_1['Lambda'], df_phe_1, 
             color='red', linewidth=0.5)

ax_left.fill_between(df_1['Lambda'], df_phe_1 - err_phe_1, df_phe_1 + err_phe_1,
                     label='0V', color='crimson', alpha=0.5)

ax_left.plot(df_2['Lambda'], df_phe_2, 
             color='blue', linewidth=0.5)

ax_left.fill_between(df_2['Lambda'], df_phe_2 - err_phe_2, df_phe_2 + err_phe_2,
                     label='900V', color='navy', alpha=0.5)

ax_left.set_xlabel('Wavelength', fontsize=15)
ax_left.set_ylabel(r'$\gamma$ / e$^-$ (A.U.)', fontsize=15)
ax_left.tick_params(axis='both', which='major', labelsize=15)
ax_left.grid()
ax_left.legend(fontsize=12)
ax_left.set_title('0V vs 900V', fontsize=14)

# --- PLOT DERECHO: resta_2 ---
ax_right = axs[1]

ax_right.plot(df_2['Lambda'], resta_2, 
              color='dodgerblue', linewidth=0.5)

ax_right.fill_between(df_2['Lambda'], resta_2 - err_resta_phe, resta_2 + err_resta_phe,
                      label='900V - 0V', color='cyan', alpha=0.5)

ax_right.set_xlabel('Wavelength', fontsize=15)
ax_right.set_ylabel(r'$\Delta\gamma$ / e$^-$ (A.U.)', fontsize=15)
ax_right.tick_params(axis='both', which='major', labelsize=15)
ax_right.grid()
ax_right.legend(fontsize=12)
ax_right.set_title('Difference (900V - 0V)', fontsize=14)

# --- GUARDAR Y MOSTRAR ---
plt.tight_layout()

plt.savefig(f'ArPurewithField.jpg', format='jpg', 
            bbox_inches='tight', dpi=300) 

plt.show(block=False)
plt.pause(0.1)
input("Presiona ENTER para cerrar la figura...")
plt.close('all')
