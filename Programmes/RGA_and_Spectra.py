#%%

import os
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np
from glob import glob
from Functions import extraer_presion, Sep_rut, Excel_value, RP, integral
import pandas as pd
from datetime import timedelta

Mother_path = r'C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\RGA'
filename = 'ArN2_99-9_0-1'

Ruta = rf"{Mother_path}\{filename}.txt"

f_hour=[1,40,00]
s_hour=[4,50,00]

number_ticks=10

names = ['Time','Nitrogene', 'Oxygen', 'Water', 'Argon', 'CF4', 'None']
rga_data = pd.read_csv(Ruta, sep=',', skiprows=27, 
                    header=None, engine='python', 
                    names=names)

# Convert Time (assumed in seconds) to timedelta objects
time_seconds = rga_data['Time']
time_labels = [str(timedelta(seconds=int(t))) for t in time_seconds]

# Hour Limits in format hh:mm:ss

f_value=3600*f_hour[0]+60*f_hour[1]+f_hour[2]
s_value=3600*s_hour[0]+60*s_hour[1]+s_hour[2]


# ---------------------------------------------- DATA ------------------------------------------------------------------------

err_SC_standard = 0.05e-7

# ----------- Reference --------------

Concentracion_N2=0.1

excel_path = r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\Whole_Data.xlsx"
base_path = rf'C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\N2\0-1'
pattern = os.path.join(base_path, '*_bar', '40kV40mA', '0V')

rutas = glob(pattern)
rutas_ordenadas = sorted(rutas, key=extraer_presion)

# ------ Reference --------

Ruta_Candela = r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\CF4\5\5_bar\40kV40mA\After_WindowChange"

Candela_FD = RP(Ruta_Candela)
df_candela = Candela_FD['calibratedResults']

filters_candela = {
    'Element A': 'Ar',
    'Concentration A':  95,
    'Element B': 'CF4',
    'Concentration B': 5,
    'Pressure (bar)': 5
}

Current_candela = Excel_value(excel_path, filters_candela, 'SC')

# -------------------------------------------------------

data_sets = []
pressures_sets = []
titles = []

pressures = []
currents = []
voltages = []
electrons = []
data = []

title = f'Ar/N2 {100-Concentracion_N2}/{Concentracion_N2}'

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

    if Excel_value(excel_path, filters, 'C3kV')==0:
       print('Current at 3kV do not exist. Using standard error of', err_SC_standard)
       err_SC = err_SC_standard

    else:
       err_SC = Excel_value(excel_path, filters, 'Err SC')
       print('Current', Excel_value(excel_path, filters, 'C3kV'))
       print('Current Error: ', err_SC)

    filtered_data = RP(i)
    df_results = filtered_data['calibratedResults']

    N_e = Saturation_current / (-1.602176634e-19)
    err_NumeroElectrones = err_SC/ (-1.602176634e-19)

    df_phe = df_results['Counts']/N_e
    err_phe = np.sqrt((df_results['Err_Counts']/N_e)**2+(df_results['Counts']*err_NumeroElectrones/(N_e)**2)**2)

    pressures.append(Presion)
    currents.append(Saturation_current)
    voltages.append(Saturation_volt)
    electrons.append(N_e)

    df = pd.DataFrame({
            'Lambda': df_results['Lambda'],
            'Counts': df_results['Counts'],
            'Err_Counts': np.sqrt(df_results['Counts'].clip(lower=0)),
            'Counts_norm': df_results['Counts']/max(df_results['Counts']),
            'Err_Counts_norm': np.sqrt(df_results['Counts'].clip(lower=0))/max(df_results['Counts']),
            'Phe':df_phe,
            'Err_Phe': err_phe
            })

    data.append(df)

    # [0]: Corriente de Saturacion
    # [1]: Voltaje de Saturacion
    # [2]: Numero de Electrones
    # [3]: Array photons per electron

data_sets.append(data)
pressures_sets.append(pressures)
titles.append(title)



# Elegí el índice del set que quieras graficar (por ejemplo, el cuarto)
idx = 0  # Cambiá este índice según la mezcla deseada

data_i = data_sets[idx]
pressures = pressures_sets[idx]
title = titles[idx]

Element_rga = ['Nitrogene', 'Oxygen', 'Water', 'Argon', 'CF4']
color_element= ['red', 'blue', 'green', 'lightblue', 'black']

# Crear figura con 2 filas y 1 columna
fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(16, 8), sharex=False, sharey=False)

# === GRÁFICO SUPERIOR ===

indice=0
for i in Element_rga: 
    ax_top.plot(time_seconds, rga_data[i]*1.33322, 
            color=color_element[indice],label=i, linewidth=2.5)
    indice+=1

ax_top.set_xticks(time_seconds[::len(time_seconds)//number_ticks],  # reduce number of ticks
        time_labels[::len(time_labels)//number_ticks],    # format those ticks
        rotation=0)
ax_top.set_title('Ar/N2 99.9/0.1', fontsize=12)
ax_top.set_xlabel('Time (hh:mm:ss)', fontsize=15)
ax_top.set_ylabel('Pressure (mBar)', fontsize=15)
ax_top.legend(fontsize=15, loc='lower right')
ax_top.tick_params(axis='both', which = 'major' ,labelsize=15)
ax_top.grid(True)
ax_top.set_xlim(f_value,s_value)
ax_top.semilogy()


# === GRÁFICO INFERIOR: espectros por presión ===
colormap = cm.turbo
norm = Normalize(vmin=min(pressures), vmax=max(pressures))

for i in range(len(pressures)):
    color = colormap(norm(pressures[i]))
    df = data_i[i]
    
    ax_bottom.plot(df['Lambda'], df['Phe'],
                   color=color,
                   linewidth=0.8,
                   label=f'{pressures[i]} bar')

    ax_bottom.fill_between(df['Lambda'],
                           df['Phe'] - df['Err_Phe'],
                           df['Phe'] + df['Err_Phe'],
                           color=color,
                           alpha=0.4)

ax_bottom.set_title(title, fontsize=12)
ax_bottom.set_xlabel(r'$\lambda$ (nm)', fontsize=11)
ax_bottom.set_ylabel(r'Photons / electrons (A.U)', fontsize=11)
ax_bottom.tick_params(axis='both', which='major', labelsize=10)
ax_bottom.grid(True)

# === Colorbar para el gráfico inferior ===
sm = cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax_bottom, label='Pressure (bar)', pad=0.02)

plt.tight_layout()
plt.savefig('2plots_espectros.jpg', dpi=300, bbox_inches='tight')
plt.show()