#%%
import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math as math
from Functions import RP, wavelength_to_rgb, Sep_rut, Excel_value

Ruta_Ar = r'C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\Ar\100\1_bar\40kV40mA\0V'
Ruta_2 = r'C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\N2\1\1_bar\40kV40mA\0V'

df_Ar=RP(Ruta_Ar)['calibratedResults']

pattern_data_2 = os.path.join(Ruta_2, 'Analized_Data', '*_AllData.txt')
archivos_2 = glob(pattern_data_2)
rut_2 = archivos_2[0]
df_2 = pd.read_csv(rut_2, sep='\t', engine='python')


Element, Concentracion, Presion, Volt_Amp, Ar_Concentration = Sep_rut(Ruta_2)

filters = {
    'Element A': 'Ar',
    'Concentration A':  Ar_Concentration,
    'Element B': Element,
    'Concentration B': Concentracion,
    'Pressure (bar)': Presion
}

Saturation_current = Excel_value(filters, 'SC')

# Crear colores del espectro visible
visible_wavelengths = np.linspace(200, 800, 1000)
colors = [wavelength_to_rgb(wl) for wl in visible_wavelengths]

# Plot
fig, ax = plt.subplots(figsize=(16, 9))

# Fondo con espectro visible
ax.imshow([colors],
          extent=[200, 800, 0, df_Ar['Phe'].max()*1.1],
          aspect='auto')

# Graficar datos con banda de error
ax.plot(df_Ar['Lambda'], df_Ar['Counts'],  
         label=f'Viejo',
         color='navy', linewidth=0.5)

ax.fill_between(df_Ar['Lambda'], 
                df_Ar['Counts'] - df_Ar['Err_Phe'],
                df_Ar['Counts'] + df_Ar['Err_Phe'],
                color='navy',
                alpha=1)

ax.plot(df_2['Lambda'], df_2['Counts'],  
         label=f'Nuevo',
         color='crimson', linewidth=0.5)

ax.fill_between(df_2['Lambda'], 
                df_2['Counts'] - df_2['Err_Phe'],
                df_2['Counts'] + df_2['Err_Phe'],
                color='crimson',
                alpha=1)

# Estética
ax.set_xlim(200, 800)
ax.set_ylim(0,)
ax.set_xlabel('wavelenght (nm)', fontsize=14)
ax.set_ylabel('Cuentas (a.u.)', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True)

plt.tight_layout()

plt.savefig(f'Corrimiento.jpg', format='jpg', 
            bbox_inches='tight', dpi = 300) 

plt.show()
