#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path_datos = r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\Dataframe_Spectra_ArCF4"
datos = pd.read_pickle(path_datos)

print(datos)

#%%

mask1 = datos['Concentrations'] == 'Ar_99_CF4_1' 
mask2 = datos['Pressure'] == 5.0
mask3 = datos['Tube Intensity'] == '40kV40mA'
mask4 = datos['Voltages'] == 0
fila = datos[mask1 & mask2 & mask3 & mask4]['Data']

corriente_pablo = datos[mask1 & mask2 & mask3 & mask4]['Currents'].values[0]

print(corriente_pablo)

#%%

wave_pablo = fila.array[0][:,0]
inte_pablo = fila.array[0][:,1]

inte_pablo_norm = inte_pablo / max(inte_pablo)

#%%

Ruta=r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\N2\1\4_bar\40kV40mA\0V"
