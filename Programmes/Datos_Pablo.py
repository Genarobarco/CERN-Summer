#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


path_datos = r"C:\Users\genar\Documents\CERN Summer 2025\Spectrum\Dataframe_Spectra_ArCF4"
datos = pd.read_pickle(path_datos)

mask1 = datos['Concentrations'] == 'N2_100' 
mask2 = datos['Pressure'] == 1
mask3 = datos['Tube Intensity'] == '40kV40mA'
mask4 = datos['Voltages'] == 0
fila = datos[mask1 & mask2 & mask3 & mask4]['Data']

wave_pablo = fila.array[0][:,0]
inte_pablo = fila.array[0][:,1]

inte_pablo /= max(inte_pablo)

plt.plot(wave_pablo, inte_pablo, label = 'pablo')

plt.legend()
plt.ylim(-0.5, 1.3)