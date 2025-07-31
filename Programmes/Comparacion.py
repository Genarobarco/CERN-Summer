#%%
import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math as math
from Functions import RP, wavelength_to_rgb, Sep_rut, Excel_value

Ruta_1 = r'C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\N2\20\5_bar\40kV40mA\0V'
Ruta_2 = r'C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\N2\20\5_bar\40kV40mA\0V_2'
Ruta_3 = r'C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\N2\20\5_bar\40kV40mA\NO_detection'

pattern_data_1 = os.path.join(Ruta_1, 'Analized_Data', '*_AllData.txt')
archivos_1 = glob(pattern_data_1)
rut_1 = archivos_1[0]
df_1 = pd.read_csv(rut_1, sep='\t', engine='python')

pattern_data_2 = os.path.join(Ruta_2, 'Analized_Data', '*_AllData.txt')
archivos_2 = glob(pattern_data_2)
rut_2 = archivos_2[0]
df_2 = pd.read_csv(rut_2, sep='\t', engine='python')

pattern_data_3 = os.path.join(Ruta_3, 'Analized_Data', '*_AllData.txt')
archivos_3 = glob(pattern_data_3)
rut_3 = archivos_3[0]
df_3 = pd.read_csv(rut_3, sep='\t', engine='python')

Element, Concentracion, Presion, Volt_Amp, Ar_Concentration = Sep_rut(Ruta_3)

filters = {
    'Element A': 'Ar',
    'Concentration A':  Ar_Concentration,
    'Element B': Element,
    'Concentration B': Concentracion,
    'Pressure (bar)': Presion
}

Saturation_current = Excel_value(filters, 'SC')

# # Crear colores del espectro visible
# visible_wavelengths = np.linspace(200, 800, 1000)
# colors = [wavelength_to_rgb(wl) for wl in visible_wavelengths]

# Plot
fig, ax = plt.subplots(figsize=(16, 9))

# Fondo con espectro visible
# ax.imshow([colors],
#           extent=[200, 800, 0, df_1['Phe'].max()*1.1],
#           aspect='auto')

# Graficar datos con banda de error
ax.plot(df_1['Lambda'], df_1['Phe'],  
         label=f'Viejo',
         color='navy', linewidth=0.5)

ax.fill_between(df_1['Lambda'], 
                df_1['Phe'] - df_1['Err_Phe'],
                df_1['Phe'] + df_1['Err_Phe'],
                color='navy',
                alpha=1)

ax.plot(df_2['Lambda'], df_2['Phe'],  
         label=f'Nuevo',
         color='crimson', linewidth=0.5)

ax.fill_between(df_2['Lambda'], 
                df_2['Phe'] - df_2['Err_Phe'],
                df_2['Phe'] + df_2['Err_Phe'],
                color='crimson',
                alpha=1)

ax.plot(df_3['Lambda'], df_3['Phe'],  
         label=f'Nuevo',
         color='green', linewidth=0.5)

ax.fill_between(df_3['Lambda'], 
                df_3['Phe'] - df_3['Err_Phe'],
                df_3['Phe'] + df_3['Err_Phe'],
                color='green',
                alpha=1)

# Estética
ax.set_xlim(200, 800)
ax.set_ylim(0,)
ax.set_xlabel('Longitud de onda (nm)', fontsize=14)
ax.set_ylabel('Cuentas (a.u.)', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True)

plt.tight_layout()

plt.savefig(f'Corrimiento.jpg', format='jpg', 
            bbox_inches='tight', dpi = 300) 

plt.show()
