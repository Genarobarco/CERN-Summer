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

Element_rga = ['Nitrogene', 'Oxygen', 'Water', 'Argon', 'CF4']
color_element= ['red', 'blue', 'green', 'lightblue', 'black']

# Crear figura con 2 filas y 1 columna
fig, ax_top = plt.subplots(figsize=(16, 9))

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
ax_top.legend(fontsize=15, loc='upper right')
ax_top.tick_params(axis='both', which = 'major' ,labelsize=15)
ax_top.grid(True)
ax_top.set_xlim(f_value,s_value)
ax_top.semilogy()
ax_top.set_ylim(1e-10)

plt.tight_layout()
plt.savefig('2plots_espectros.jpg', dpi=300, bbox_inches='tight')
plt.show()