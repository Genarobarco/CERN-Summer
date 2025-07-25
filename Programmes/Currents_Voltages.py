#%%

import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math as math
from glob import glob
from Functions import Excel_value, RP, data_pablo

df_pablo = data_pablo('Ar_95_CF4_5', 5.0, '40kV40mA', 0)

R_1 = r'C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\N2\0-5\1_bar\40kV40mA\max_collection'
R_1_MC = r'C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\N2\0-5\1_bar\40kV40mA\0V'

R_2 = r'C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\N2\0-5\2_bar\40kV40mA\max_collection'
R_2_MC = r'C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\N2\0-5\2_bar\40kV40mA\0V'

R_3 = r'C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\N2\0-5\3_bar\40kV40mA\max_collection'
R_3_MC = r'C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\N2\0-5\3_bar\40kV40mA\0V'

R_4 = r'C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\N2\0-5\4_bar\40kV40mA\max_collection'
R_4_MC = r'C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\N2\0-5\4_bar\40kV40mA\0V'

R_5 = r'C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\N2\0-5\5_bar\40kV40mA\max_collection'
R_5_MC = r'C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\N2\0-5\5_bar\40kV40mA\0V'

rutas_0V=[R_1, R_2, R_3, R_4, R_5]
rutas_MC=[R_1_MC, R_2_MC, R_3_MC, R_4_MC, R_5_MC]
presiones = [1,2,3,4,5]

pares = []

for i in range(5):

    filters = {
        'Element A': 'Ar',
        'Concentration A':  99.5,
        'Element B': 'N2',
        'Concentration B': 0.5,
        'Pressure (bar)': presiones[i]
        }
    
    df=RP(rutas_0V[i])['calibratedResults']
    df_MC=RP(rutas_MC[i])['calibratedResults']
    Sc = Excel_value(filters, 'SC')
    Sv = Excel_value(filters, 'SV')

    df_phe = df['Counts']

    pares.append((df, df_MC, Sc, Sv))

# Crear figura con 2x3 subplots
fig, axs = plt.subplots(2, 3, figsize=(16, 9), sharex=False)

# Compartir eje Y por fila
axs[0, 1].sharey(axs[0, 0])
axs[0, 2].sharey(axs[0, 0])
axs[1, 0].sharey(axs[1, 0])
axs[1, 1].sharey(axs[1, 0])

# Colores para los espectros
color_original = 'darkorange'
color_MC = 'navy'

# Listas para Sc y Sv
Sc_list = []
Sv_list = []

# Primeros 5 subplots: espectros
for idx, (df, df_MC, Sc, Sv) in enumerate(pares):
    row = idx // 3
    col = idx % 3
    ax = axs[row, col]

    # Guardar valores para el plot final
    Sc_list.append(Sc)
    Sv_list.append(Sv)

    # Plot original
    ax.plot(df['Lambda'], df['Counts'].clip(lower=0), color=color_original, linewidth=0.8, label='0V')
    ax.fill_between(df['Lambda'],
                    df['Counts'].clip(lower=0) - df['Err_Counts'],
                    df['Counts'].clip(lower=0) + df['Err_Counts'],
                    color=color_original, alpha=0.3)

    # Plot Monte Carlo
    ax.plot(df_MC['Lambda'], df_MC['Counts'].clip(lower=0), color=color_MC, linewidth=0.8, label=f'{Sv}V')
    ax.fill_between(df_MC['Lambda'],
                    df_MC['Counts'].clip(lower=0) - df_MC['Err_Counts'],
                    df_MC['Counts'].clip(lower=0) + df_MC['Err_Counts'],
                    color=color_MC, alpha=0.3)

    ax.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=9)

    if row == 1:
        ax.set_xlabel(r'$\lambda$ (nm)', fontsize=10)
    if col == 0:
        ax.set_ylabel(r'Photons / electrons (A.U)', fontsize=10)

    ax.set_title(f'Pressure: {presiones[idx]} bar', fontsize=10)
    ax.legend(fontsize=8)

# === Sexto subplot: SC y SV vs presión ===
ax_sc = axs[1, 2]              # Eje principal
ax_sv = ax_sc.twinx()          # Segundo eje Y

# SC en azul
ax_sc.plot(presiones, Sc_list, 'o-', color='red', label='SC (A)', linewidth=2)
ax_sc.set_ylabel('Saturation Current (A)', color='red', fontsize=10)
ax_sc.tick_params(axis='y', labelcolor='red')

# SV en rojo
ax_sv.plot(presiones, Sv_list, 's--', color='navy', label='SV (V)', linewidth=2)
ax_sv.set_ylabel('Saturation Voltage (V)', color='navy', fontsize=10)
ax_sv.tick_params(axis='y', labelcolor='navy')

ax_sc.set_xlabel('Pressure (bar)', fontsize=10)
ax_sc.set_title('SC and SV vs Pressure', fontsize=10)
ax_sc.grid(True)

# Ajuste layout final
fig.subplots_adjust(hspace=0.15, wspace=0.2, left=0.06, right=0.94)

# Guardar y mostrar
plt.savefig('Comparacion_Mediciones_ConSinCampo.jpg', dpi=300, bbox_inches='tight')
plt.show(block=False)
plt.pause(0.1)
input("Presiona ENTER para cerrar la figura...")
plt.close('all')
