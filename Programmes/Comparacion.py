#%%

import sys
import os
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from scipy.stats import norm
import numpy as np
import pandas as pd
import math as math
from Functions import extraer_presion, integral

def cargar_datos(rutas):
    datos = []
    presiones = []

    for ruta in rutas:
        pattern_data = os.path.join(ruta, 'Analized_Data', '*_AllData.txt')
        archivos = glob(pattern_data)

        if not archivos:
            print(f"No se encontró archivo en: {pattern_data}")
            continue

        df = pd.read_csv(archivos[0], sep='\t', engine='python')
        datos.append(df)

        # Si corresponde, extraemos presión
        presiones.append(extraer_presion(ruta))

    return datos, presiones

element_mix = 'N2'
Concentration_mix = 20

yield_factor = .15
err_lambda = 1 #nm
integration_limits = [300,450] #nm


# --------------- Plot Diferencia --------------------------


path_NO = rf'C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\{element_mix}\{Concentration_mix}'
pattern_NO = os.path.join(path_NO, '5_bar', '40kV40mA', 'max_collection')

rutas_NO = glob(pattern_NO)
NO_ruts = sorted(rutas_NO, key=extraer_presion)

path = rf'C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\{element_mix}\{Concentration_mix}'
pattern = os.path.join(path, '5_bar', '40kV40mA', '0V')

rutas = glob(pattern)
ruts = sorted(rutas, key=extraer_presion)

data, presiones = cargar_datos(ruts)
data_NO, _ = cargar_datos(NO_ruts) 

df = data[0]
df_NO = data_NO[0]


color_clean = 'crimson'
color_dirty = 'navy'

plt.figure(figsize=(16, 9))

plt.plot(df['Lambda'], df['Phe'] - df_NO['Phe'], color=color_clean, label='BakeOut and Pumped')
# plt.fill_between(df['Lambda'], 
#                 df['Phe'] - df['Err_Phe'], 
#                 df['Phe'] + df['Err_Phe'], 
#                 color=color_clean, alpha=0.3)

# plt.plot(df_NO['Lambda'], df_NO['Phe'], color=color_dirty, label='Plastic Pipe Conected')
# plt.fill_between(df_NO['Lambda'], 
#                 df_NO['Phe'] - df_NO['Err_Phe'], 
#                 df_NO['Phe'] + df_NO['Err_Phe'], 
#                 color=color_dirty, alpha=0.3)

# plt.set_ylim(-0.1,20)
plt.grid(True)
plt.legend(fontsize = 15)
plt.title('Comparison Pure N2 Contaminated', fontsize = 15)
plt.tick_params(axis='both', which = 'major', labelsize = 15)
plt.xlabel('wavelength (nm)', fontsize = 15)
plt.ylabel('Photons / electrons (A.U.)', fontsize = 15)

plt.tight_layout()
plt.savefig(f"ArN2_{100 - Concentration_mix}{Concentration_mix}-0V_MaxCollection.jpg", dpi=300, bbox_inches='tight')
plt.show(block=False)
plt.pause(0.1)
input("Presiona ENTER para cerrar la figura...")
plt.close('all')

