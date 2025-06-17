#%% Library Cell

import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math as math
from glob import glob

#%%

Rutas=glob(r'C:\Users\genar\Documents\CERN Summer 2025\BG Studies\100 ms\*.txt')

data={}
Count_List=[]
Indice= 0

for ruta in Rutas:

  data_list=[]

  #Caracterizacion

  # As the files to be used are made of two columbs one for the wave length (Lambda) and other for the
  # intensiti of each Lambda, here we just have to set another name in the array variable and uncomment the
  # line 'Colum Lambda

  array= pd.read_csv(ruta, names=['Lambda', 'Counts'], sep='\t', 
                     skiprows=17, skipfooter=1, header=None, engine='python')

  Colum_Cuentas=array['Counts']
  Colum_Lambda=array['Lambda'] 

  Count_List+=[Colum_Cuentas]
  data_list+=[Colum_Lambda, Colum_Cuentas]
  data[Indice]=data_list

  Indice += 1


channel_length=len(data[0][0])
mean_per_channel=[]
var_per_channel=[]

for j in np.arange(0,2048):

  lista=[]

  for i in np.arange(0,len(Count_List)):

    lista.append(Count_List[i][j])
    
  mean_per_channel.append(np.mean(lista))
  var_per_channel.append(np.var(lista))


channels=data[0][0]

plt.figure(figsize=(12,8))
plt.errorbar(channels, mean_per_channel, fmt='.', 
             color='mediumorchid', yerr=var_per_channel,
             ecolor='teal', elinewidth=0.5, 
             capsize=5, capthick=0.5, 
            errorevery=1, alpha=0.9)
plt.xlabel('Wave Lenght (nm)', fontsize=20)
plt.ylabel('Intensity (A.U.)', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.title('100 ms histogram', fontsize=20)
plt.ylim(-1750,2250)
plt.savefig('100 ms Mean Histogram.jpg', format='jpg',
            bbox_inches='tight')


