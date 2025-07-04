#%% Library Cell

import os
import re
from glob import glob
import numpy as np
import pandas as pd
import DataSorting as DS
from DataSorting import landau, gaussian

Rutas=glob(r'C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\CF4\100\40kV40mA\1_test\DataBG\Data0.txt')

mean_per_channel = DS.Reader(Rutas)[0]
std_per_channel = DS.Reader(Rutas)[1]

print(DS.Reader(Rutas)[2])


DS.Histo(mean_per_channel, landau, [100,100,1], 500, 
      'Counts BG', color = 'blue', lim = True, 
      x_min= -1000, x_max=5000)

# DS.Histo(std_per_channel, gaussian, [50,20,1], 1000, 'STD BG', 
#       color = 'orange', lim = True, x_min= -10, x_max=120)