#%%

import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math as math
from glob import glob
from scipy.stats import norm
from scipy.optimize import curve_fit

import matplotlib.cm as cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

Rutas_drift=glob(r'C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\CF4\1\5_bar\40kV40mA\test_pure\DataBG\*.txt')

data_drift={}
Count_List_drift=[]
Indice_drift= 0

for ruta in Rutas_drift:

  data_drift_list=[]

  array= pd.read_csv(ruta, names=['Lambda', 'Counts'], sep='\t', 
                     skiprows=17, skipfooter=1, header=None, engine='python')

  Colum_Cuentas_drift=array['Counts']
  Colum_Lambda_drift=array['Lambda'] 

  Count_List_drift+=[Colum_Cuentas_drift]
  data_drift_list+=[Colum_Lambda_drift, Colum_Cuentas_drift]
  data_drift[Indice_drift]=data_drift_list

  Indice_drift += 1

# Plotting
fig, ax = plt.subplots(figsize=(12,8))

cmap = plt.get_cmap('inferno')
num_plots = len(data_drift)

colors = [cmap(i / num_plots) for i in range(num_plots)]

for idx in range(len(data_drift)):
    ax.plot(data_drift[0][0], data_drift[idx][1], lw=0.5, color=colors[idx])

ax.set_xlabel('Wave Lenght (nm)', fontsize=20)
ax.set_ylabel('Intensity (A.U.)', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=18)
ax.set_title('1000 ms BG drift', fontsize=15)

# Create colorbar
norm = Normalize(vmin=0, vmax=num_plots)
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('Data set', fontsize=15)

plt.savefig('1000 ms BG Drift.jpg', format='jpg',
            bbox_inches='tight')

plt.show()