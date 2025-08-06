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
Concentration_mix = 50

yield_factor = .15
err_lambda = 1 #nm
integration_limits = [300,450] #nm


# --------------- Plot Diferencia --------------------------


path_NO = rf'C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\{element_mix}\{Concentration_mix}'
pattern_NO = os.path.join(path_NO, '*_bar', '40kV40mA', '0V_withNO')

rutas_NO = glob(pattern_NO)
NO_ruts = sorted(rutas_NO, key=extraer_presion)

path = rf'C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\{element_mix}\{Concentration_mix}'
pattern = os.path.join(path, '*_bar', '40kV40mA', '0V')

rutas = glob(pattern)
ruts = sorted(rutas, key=extraer_presion)

dicc_dataframes = {}
presiones = []
data = []
data_NO =[]
integrales = []
integrales_err = []
integrales_NO =[]
integrales_NO_err = []

data, presiones = cargar_datos(ruts)
data_NO, _ = cargar_datos(NO_ruts) 

print(presiones)


for ValPressureIndex in range(len(presiones)):
    sum, err = integral(data[ValPressureIndex], 
                    'Lambda', err_lambda,
                    'Phe', 'Err_Phe',
                    *integration_limits)
    integrales.append(sum)
    integrales_err.append(err)

    sum_NO, err_NO = integral(data_NO[ValPressureIndex], 
                    'Lambda', err_lambda,
                    'Phe', 'Err_Phe',
                    *integration_limits)
    integrales_NO.append(sum_NO)
    integrales_NO_err.append(err_NO)


# Crear figura 2x3 sin compartir ejes
fig, axs = plt.subplots(2, 3, figsize=(15, 8))

# Reemplazar el sexto eje (axs[1,2]) por uno nuevo sin compartir ejes
fig.delaxes(axs[1, 2])  # Borrar el original
ind_ax = fig.add_subplot(2, 3, 6)  # Posición 6 → fila 2, col 3
axs[1, 2] = ind_ax  # Asignar el nuevo eje

# Aplanar para facilidad
axs = axs.flatten()

color = 'crimson'
color_NO = 'navy'

# Graficar los primeros 5 datasets (índices 0 a 4) con ejes compartidos
for i in range(5):
    ax = axs[i]
    df = data[i]
    df_NO = data_NO[i]

    ax.plot(df['Lambda'], df['Phe'], color=color, label='clean')
    ax.fill_between(df['Lambda'], 
                    df['Phe'] - df['Err_Phe'], 
                    df['Phe'] + df['Err_Phe'], 
                    color=color, alpha=0.3)
    
    ax.plot(df_NO['Lambda'], df_NO['Phe'], color=color_NO, label='Contaminated')
    ax.fill_between(df_NO['Lambda'], 
                    df_NO['Phe'] - df_NO['Err_Phe'], 
                    df_NO['Phe'] + df_NO['Err_Phe'], 
                    color=color_NO, alpha=0.3)

    ax.set_ylim(-0.1,20)
    ax.set_title(f'{presiones[i]} bar')
    ax.grid(True)
    ax.legend()

    if i >= 3:
        ax.set_xlabel(r'$\lambda$ (nm)')
    if i % 3 == 0:
        ax.set_ylabel('Photons / electrons (A.U)')

# Sexto subplot independiente (posición 5)

integrales = np.array(integrales)
integrales_NO = np.array(integrales_NO)

ax6 = axs[5]
ax6.set_title("Scintillation performance")

ax6.errorbar(presiones, integrales, yerr=integrales*yield_factor, fmt='o:',
             color=color, label='0V')
ax6.errorbar(presiones, integrales_NO, yerr=integrales_NO*yield_factor, fmt='o:',
             color=color_NO, label='Max Collection')

ax6.set_xlabel('Pressure (bar)')
ax6.set_ylabel(r'$\int \gamma / e^- \cdot \lambda$')
ax6.grid(True)
ax6.legend()

# Ajustar layout
fig.suptitle(f"Ar/N2 {100 - Concentration_mix}/{Concentration_mix} - Comparison 0V v Max Collection", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])  # deja espacio para el título

plt.savefig(f"ArN2_{100 - Concentration_mix}{Concentration_mix}-0V_MaxCollection.jpg", dpi=300, bbox_inches='tight')
plt.show(block=False)
plt.pause(0.1)
input("Presiona ENTER para cerrar la figura...")
plt.close('all')

