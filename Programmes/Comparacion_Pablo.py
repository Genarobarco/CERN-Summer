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
from Functions import extraer_presion, integral, data_pablo


Ruta_N2_withNO = r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\N2\100\5_bar\40kV40mA\0V_wihNO\Analized_Data\ArN2_0100_5_AllData.txt"

NON2 = pd.read_csv(Ruta_N2_withNO, sep='\t', header=0)

df_pablo, current_pablo = data_pablo('Ar_99_CF4_1', 5, '40kV40mA', 0)

print(df_pablo)
#%%

plt.figure(figsize=(16,9))

plt.plot(df_pablo['Lambda'], df_pablo['Counts'], label='Sunday')
plt.plot(NON2['Lambda'], NON2['Phe'], label = 'Friday')

plt.legend(fontsize=20)
plt.xlabel('Wavelenght', fontsize=15)
plt.ylabel(r'$\gamma$ / e$^-$ (A.U.)', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.grid()
plt.show(block=False)
plt.pause(0.1)
input("Press enter to close all figures...")
plt.close('all')