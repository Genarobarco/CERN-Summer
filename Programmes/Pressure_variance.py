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
from Functions import Sep_rut, RP, Excel_value, extraer_presion, integral

# ------ Errors -----------

err_SC_standard = 0.05e-7
err_SV = 50 #v
err_lambda = 5 #nm
err_pressure = 10e-3 #bar

# ----------- Integration Limits -----------------

integration_limits = [300, 450]
integration_limits_ref = [500, 750]

# -------- element and concentration ------------s
element_mix = 'N2'
Concentracion_N2 = 5



if '.' in str(Concentracion_N2):
   Concentration_mix = str(Concentracion_N2).replace('.','-')

else:
   Concentration_mix = Concentracion_N2

# -------- Paths ------------

base_path = rf'C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\{element_mix}\{Concentration_mix}'
pattern = os.path.join(base_path, '*_bar', '40kV40mA', '0V')

rutas = glob(pattern)
rutas_ordenadas = sorted(rutas, key=extraer_presion)


# ------ Reference --------

Ruta_Candela = r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\CF4\5\5_bar\40kV40mA\After_WindowChange"

pattern_data_ref = os.path.join(Ruta_Candela, 'Analized_Data', '*_AllData.txt')
archivos_ref = glob(pattern_data_ref)

rut_ref = archivos_ref[0]
df_ref = pd.read_csv(rut_ref, sep='\t', engine='python')

filters_candela = {
    'Element A': 'Ar',
    'Concentration A':  95,
    'Element B': 'CF4',
    'Concentration B': 5,
    'Pressure (bar)': 5
}

Current_candela = Excel_value(filters_candela, 'SC')

NumElectronsCandela = Current_candela / (-1.602176634e-19)
err_NumE_candela = err_SC_standard/ (-1.602176634e-19)

sum_ref, err_int_ref = integral(df_ref, 
                    'Lambda', err_lambda,
                    'Phe', 'Err_Phe', *integration_limits_ref)

#%%

# -------------------------------------------------------

pressures = []
currents = []
voltages = []
data = []
data_integral=[]


for i in rutas_ordenadas:

    Element, Concentracion, Presion, Volt_Amp, Ar_Concentration = Sep_rut(i)

    filters = {
        'Element A': 'Ar',
        'Concentration A':  Ar_Concentration,
        'Element B': Element,
        'Concentration B': Concentracion,
        'Pressure (bar)': Presion
    }

    Saturation_current = Excel_value(filters, 'SC')
    Saturation_volt = Excel_value(filters, 'SV')

    if Excel_value(filters, 'C3kV')==0:
       print('Current at 3kV do not exist. Using standard error of', err_SC_standard)
       err_SC = err_SC_standard

    else:
       Saturation_current_3kV = Excel_value(filters, 'C3kV')
       err_SC  = abs(Saturation_current_3kV - Saturation_current)
       print('Current:', Saturation_current)
       print('Current at 3kV:', Saturation_current_3kV)
       print('Current Error: ', err_SC)

    pattern_data = os.path.join(i, 'Analized_Data', '*_AllData.txt')
    archivos = glob(pattern_data)
    
    if archivos:
        archivo = archivos[0]
        df = pd.read_csv(archivo, sep='\t', engine='python')
    else:
        print(f"No se encontró archivo en {i}")

    N_e = Saturation_current / (-1.602176634e-19)
    err_NumeroElectrones = err_SC/ (-1.602176634e-19)

    pressures.append(Presion)
    currents.append(Saturation_current)
    voltages.append(Saturation_volt)
    data.append(df)

    # [0]: Corriente de Saturacion
    # [1]: Voltaje de Saturacion
    # [2]: Numero de Electrones
    # [3]: Array photons per electron

print('Lista de presiones: ', pressures)
integrales = []
err_integral = []
for ValPressureIndex in range(len(pressures)):
    sum, err = integral(data[ValPressureIndex], 
                    'Lambda', err_lambda,
                    'Phe', 'Err_Phe',
                    *integration_limits)
    integrales.append(sum)
    err_integral.append(err)

df = pd.DataFrame({
        'Pressures': pressures,
        'integrals': integrales, 
        'Err_int': err_integral
        })

data_integral.append(df)

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
           bbox_to_anchor=(0.21, 0.9))

plt.savefig(f'Ar{Element}_{Ar_Concentration}{Concentracion}_PV_CV.jpg', format='jpg', 
            bbox_inches='tight', dpi = 300) 

plt.tight_layout()

#%%

# [i][j]: i-> 0,1,2,3,4,5 = Presion

norm = mcolors.Normalize(vmin=min(pressures), vmax=max(pressures))
colormap = plt.colormaps['turbo']

fig, ax = plt.subplots(figsize=(12, 8))

for i in range(len(pressures)):
    color = colormap(norm(pressures[i]))
    ax.plot(data[i]['Lambda'], data[i]['Counts'].clip(lower=0),
            label=f'{pressures[i]} bar',
            color=color,
            linewidth=0.5)
    ax.fill_between(data[i]['Lambda'], 
                    data[i]['Counts']-data[i]['Err_Counts'],
                    data[i]['Counts']+data[i]['Err_Counts'],
                    color=color,
                    alpha = 0.5)

ax.set_xlabel(r'$\lambda$ (nm)', fontsize=15)
ax.set_ylabel('Absolute Counts (A.U)', fontsize=15)
ax.tick_params(axis='both', which='major', labelsize=13)
ax.grid()
ax.legend(fontsize=13, title='Pressures', title_fontsize=12)

sm = cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])

# cbar = fig.colorbar(sm, ax=ax)
# cbar.set_label('Pressure (bar)', fontsize=13)
# cbar.ax.tick_params(labelsize=12)

plt.savefig(f'Ar{Element}_{Ar_Concentration}{Concentracion}_PV_Abs.jpg', format='jpg', 
            bbox_inches='tight', dpi = 300) 

plt.tight_layout()

#%%

norm = mcolors.Normalize(vmin=min(pressures), vmax=max(pressures))
colormap = plt.colormaps['turbo']

fig, ax = plt.subplots(figsize=(12, 8))

for i in range(len(pressures)):
    color = colormap(norm(pressures[i]))
    ax.plot(data[i]['Lambda'], data[i]['Counts_norm'].clip(lower=0),
            label=f'{pressures[i]} bar',
            color=color,
            linewidth=0.5)
    ax.fill_between(data[i]['Lambda'], 
                    data[i]['Counts_norm']-data[i]['Err_Counts_norm'],
                    data[i]['Counts_norm']+data[i]['Err_Counts_norm'],
                    color=color,
                    alpha = 0.5)

ax.set_xlabel(r'$\lambda$ (nm)', fontsize=15)
ax.set_ylabel('Normalize Counts (A.U)', fontsize=15)
ax.tick_params(axis='both', which='major', labelsize=13)
ax.grid()
ax.legend(fontsize=13, title='Pressures', title_fontsize=12)

sm = cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])

# cbar = fig.colorbar(sm, ax=ax)
# cbar.set_label('Pressure (bar)', fontsize=13)
# cbar.ax.tick_params(labelsize=12)

plt.savefig(f'Ar{Element}_{Ar_Concentration}{Concentracion}_PV_Norms.jpg', format='jpg', 
            bbox_inches='tight', dpi = 300) 

plt.tight_layout()

#%%

norm = mcolors.Normalize(vmin=min(pressures), vmax=max(pressures))
colormap = plt.colormaps['turbo']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))

# --- Primer gráfico: espectros ---
for i in range(len(pressures)):
    color = colormap(norm(pressures[i]))
    ax1.plot(data[i]['Lambda'], data[i]['Phe'], color=color, linewidth=0.5)
    ax1.fill_between(data[i]['Lambda'],
                     data[i]['Phe'] - data[i]['Err_Phe'],
                     data[i]['Phe'] + data[i]['Err_Phe'],
                     label=f'{pressures[i]} bar',
                     color=color, alpha=0.5)

ax1.fill_between(df_ref['Lambda'],
                 df_ref['Phe']- df_ref['Err_Phe'],
                 df_ref['Phe']+ df_ref['Err_Phe'],
                 label=f'Reference - {Current_candela} A',
                 color='magenta', alpha=0.5)

ax1.set_xlabel(r'$\lambda$ (nm)', fontsize=15)
ax1.set_ylabel(r'Photons / electrons (A.U)', fontsize=15)
ax1.tick_params(axis='both', which='major', labelsize=13)
ax1.grid()
ax1.legend(fontsize=10, title='Pressures', title_fontsize=11)
ax1.set_title(f'Ar/{Element} {Ar_Concentration}/{Concentracion}', fontsize=14)

# Colorbar para el primer gráfico
sm = cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax1)
cbar.set_label('Pressure (bar)', fontsize=13)
cbar.ax.tick_params(labelsize=12)

# --- Segundo gráfico: integrales ---

ax2.errorbar(df['Pressures'], df['integrals'],
                yerr=df['Err_int'], xerr=err_pressure,
                fmt='o-', label=f'{100 - Concentracion_N2}/{Concentracion_N2}',
                color=color)

ax2.set_xlabel('Pressure (bar)', fontsize=12)
ax2.set_ylabel(r'$\gamma$/e$^-$ $\times \lambda$', fontsize=12)
ax2.tick_params(axis='both', which='major', labelsize=11)
ax2.grid()
ax2.legend(fontsize=10, title='Ar/N2', title_fontsize=10)
ax2.set_title('Integrated Spectra', fontsize=14)

plt.tight_layout()
plt.savefig(f'Ar{Element}_{Ar_Concentration}{Concentracion}_PV_Phe.jpg',
            format='jpg', bbox_inches='tight', dpi=300)

plt.show(block=False)
plt.pause(0.1)
input("Press enter to close all figures...")
plt.close('all')
