#%%
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import math as math
from glob import glob
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.special import factorial

def Excel_value(file_path, filters, target_column):
    df = pd.read_excel(file_path)
    
    mask = pd.Series(True, index=df.index)
    for col, val in filters.items():
        mask &= (df[col] == val)
    
    filtered_df = df[mask]
    
    if filtered_df.empty:
        print("No match found with the given filters.")
        return None
    
    return filtered_df[target_column].values[0]

def Sep_rut(Ruta, Mother_folder = 'Spectra_2025_Pablo_Raul_Genaro'):
  partes = Ruta.split('\\')

  for i, parte in enumerate(partes):

    if parte == Mother_folder:
      indicador = i

  Element = partes[indicador+1]
  Concentracion = int(partes[indicador+2])
  Presion = partes[indicador+3].split('_')[0]
  Volt_Amp = partes[indicador+4]

  if Presion.split('_'):
    Presion = float(Presion.replace('-', '.'))

  Ar_Concentration = 100 - int(Concentracion)

  return Element, Concentracion, Presion, Volt_Amp, Ar_Concentration

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

def extraer_presion(ruta):
    """Extrae la presión en float desde la ruta."""
    partes = ruta.split(os.sep)
    for parte in partes:
        if '_bar' in parte:
            presion_str = parte.replace('_bar', '').replace('-', '.')
            try:
                return float(presion_str)
            except ValueError:
                return float('inf')  # Por si hay algún error, lo manda al final
    return float('inf')

element_mix = 'N2'
concentration_mix = 1

excel_path = r'C:\Users\genar\VSC code\CERN-Summer\Whole_Data.xlsx'
base_path = rf'C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\{element_mix}\{concentration_mix}'
pattern = os.path.join(base_path, '*_bar', '40kV40mA', '0V')

rutas = glob(pattern)
rutas_ordenadas = sorted(rutas, key=extraer_presion)

pressures = []
currents = []
voltages = []
electrons = []
data = []

for i in rutas_ordenadas:

    Element, Concentracion, Presion, Volt_Amp, Ar_Concentration = Sep_rut(i)

    filters = {
        'Element A': 'Ar',
        'Concentration A':  Ar_Concentration,
        'Element B': Element,
        'Concentration B': Concentracion,
        'Pressure (bar)': Presion
    }

    Saturation_current = Excel_value(excel_path, filters, 'SC')
    Saturation_volt = Excel_value(excel_path, filters, 'SV')

    filtered_data = RP(i)
    df_results = filtered_data['calibratedResults']

    N_e = Saturation_current / (-1.602176634e-19)
    df_phe = df_results['Counts']/N_e

    pressures.append(Presion)
    currents.append(Saturation_current)
    voltages.append(Saturation_volt)
    electrons.append(N_e)
    data.append([df_results, df_phe])

    # [0]: Corriente de Saturacion
    # [1]: Voltaje de Saturacion
    # [2]: Numero de Electrones
    # [3]: Array photons per electron

#%%
fig, ax1 = plt.subplots(figsize=(12, 8))

# First plot: Saturation Current (left y-axis)
color1 = 'navy'
ax1.errorbar(pressures, voltages, yerr=30,
             label='SV', color=color1,
             linewidth=0.5, fmt='.:', markersize=15,
             elinewidth=2, capsize=3)
ax1.set_xlabel('Pressure (bar)', fontsize=15)
ax1.set_ylabel('Saturation Voltaje (A)', fontsize=15)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.tick_params(axis='both', which='major', labelsize=15)
ax1.grid()


# Second plot: Saturation Voltage (right y-axis)
ax2 = ax1.twinx()
color2 = 'crimson'
ax2.errorbar(pressures, currents, yerr=0.05e-7,
             label='SC', color=color2,
             linewidth=0.5, fmt='.:', markersize=15,
             elinewidth=2, capsize=3)
ax2.set_ylabel('Saturation Current (V)', fontsize=15)
ax2.tick_params(axis='y', labelcolor=color2)


ax1.tick_params(axis='both', which='major', labelsize=14)
ax2.tick_params(axis='both', which='major', labelsize=14)

# Legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
fig.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=15, 
           bbox_to_anchor=(0.2, 0.95))

plt.savefig(f'Ar{Element}_{Ar_Concentration}{Concentracion}_PV_CV.jpg', format='jpg', 
            bbox_inches='tight', dpi = 300) 

plt.tight_layout()

#%%

# [i][j]: i-> 0,1,2,3,4,5 = Presion
#         j-> 0,1 = df_results, df_phe

Lambda = data[0][0]['Lambda']

norm = mcolors.Normalize(vmin=min(pressures), vmax=max(pressures))
colormap = cm.get_cmap('turbo')

fig, ax = plt.subplots(figsize=(12, 8))

for i in range(len(pressures)):
    color = colormap(norm(pressures[i]))
    ax.plot(Lambda, data[i][0]['Counts'].clip(lower=0),
            label=f'{pressures[i]} bar',
            color=color,
            linewidth=0.5)

ax.set_xlabel('Wavelength', fontsize=15)
ax.set_ylabel('Absolute Counts (A.U)', fontsize=15)
ax.tick_params(axis='both', which='major', labelsize=13)
ax.grid()
ax.legend(fontsize=13, title='Pressures', title_fontsize=12)

sm = cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('Pressure (bar)', fontsize=13)
cbar.ax.tick_params(labelsize=12)

plt.savefig(f'Ar{Element}_{Ar_Concentration}{Concentracion}_PV_Abs.jpg', format='jpg', 
            bbox_inches='tight', dpi = 300) 

plt.tight_layout()

#%%

norm = mcolors.Normalize(vmin=min(pressures), vmax=max(pressures))
colormap = cm.get_cmap('turbo')

fig, ax = plt.subplots(figsize=(12, 8))

for i in range(len(pressures)):
    color = colormap(norm(pressures[i]))
    ax.plot(Lambda, data[i][0]['Counts_norm'].clip(lower=0),
            label=f'{pressures[i]} bar',
            color=color,
            linewidth=0.5)

ax.set_xlabel('Wavelength', fontsize=15)
ax.set_ylabel('Normalize Counts (A.U)', fontsize=15)
ax.tick_params(axis='both', which='major', labelsize=13)
ax.grid()
ax.legend(fontsize=13, title='Pressures', title_fontsize=12)

sm = cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('Pressure (bar)', fontsize=13)
cbar.ax.tick_params(labelsize=12)

plt.savefig(f'Ar{Element}_{Ar_Concentration}{Concentracion}_PV_Norms.jpg', format='jpg', 
            bbox_inches='tight', dpi = 300) 

plt.tight_layout()

#%%

norm = mcolors.Normalize(vmin=min(pressures), vmax=max(pressures))
colormap = cm.get_cmap('turbo')

fig, ax = plt.subplots(figsize=(12, 8))

for i in range(len(pressures)):
    color = colormap(norm(pressures[i]))
    ax.plot(Lambda, data[i][1],
            label=f'{pressures[i]} bar',
            color=color,
            linewidth=0.5)

ax.set_xlabel('Wavelength', fontsize=15)
ax.set_ylabel(r'Photons / electrons (A.U)', fontsize=15)
ax.tick_params(axis='both', which='major', labelsize=13)
ax.grid()
ax.legend(fontsize=13, title='Pressures', title_fontsize=12)

sm = cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('Pressure (bar)', fontsize=13)
cbar.ax.tick_params(labelsize=12)

plt.savefig(f'Ar{Element}_{Ar_Concentration}{Concentracion}_PV_phe.jpg', format='jpg', 
            bbox_inches='tight', dpi = 300) 

plt.tight_layout()

plt.show(block=False)
plt.pause(0.1)
input("Press enter to close all figures...")
plt.close('all')