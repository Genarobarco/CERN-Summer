#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math as math
from Functions import RP, data_pablo, Excel_value


ConcentracionCF4 = 5
Presion = 5

ruta_N2 = r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\N2\100\5_bar\40kV40mA\0V\Analized_Data\ArN2_0100_5_AllData.txt"
ruta_N2NO = r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\N2\100\5_bar\40kV40mA\0V_wihNO\Analized_Data\ArN2_0100_5_AllData.txt"

df_N2 = pd.read_csv(ruta_N2, sep='\t', header = 0)
df_N2NO = pd.read_csv(ruta_N2NO, sep='\t', header = 0)

plt.figure(figsize=(16,9))

plt.plot(df_N2['Lambda'], df_N2['Phe'], label='Hoy')
plt.plot(df_N2NO['Lambda'], df_N2NO['Phe'], label = 'Viernes')

plt.legend(fontsize=20)
plt.xlabel('Wavelenght', fontsize=15)
plt.ylabel(r'$\gamma$ / e$^-$ (A.U.)', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.grid()
plt.show(block=False)
plt.pause(0.1)
input("Press enter to close all figures...")
plt.close('all')