#%% Library Cell

import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math as math
from glob import glob
from scipy.stats import norm
from scipy.optimize import curve_fit

#%% 

dataset_title='100 ms'

def gaussian(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) *np.exp(- (x - mu)**2 
                                                       / (2 * sigma**2))

# -------------------------- DATA SORTING ----------------------------
# Data sorting

Rutas=glob(r'C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\CF4\1\1_bar\0V\test_pure\DataBG\*.txt')

Count_List=[]
Lambda_List=[]
Indice= 0

for ruta in Rutas:

  data_list=[]

  array= pd.read_csv(ruta, names=['Lambda', 'Counts'], sep='\s+', 
                     header=None, engine='python')

  Colum_Cuentas=array['Counts']
  Colum_Lambda=array['Lambda'] 

  Count_List+=[Colum_Cuentas]
  Lambda_List+=[Colum_Lambda]

  Indice += 1


#%%

mean_per_channel=[]
std_per_channel=[]

for j in np.arange(0,2048):

  lista=[]

  for i in np.arange(0,len(Count_List)):

    lista.append(Count_List[i][j])
    
  mean_per_channel.append(np.mean(lista))
  std_per_channel.append(np.std(lista))

channels=Lambda_List[0]

mean_per_channel


#%%
# ------------------ STATISTICAL ANALYSIS ---------------------


# For each data set (exposure time sorted), must create an histogram for
# the STD variance per channel, that means, create an histogram for each
# std_per_chanel.

min_value=min(std_per_channel)
max_value=max(std_per_channel)

plt.figure(figsize=(12,8))

# Histogram configuration
data= std_per_channel
bins = 900                 # Number of bins
range_values = (min_value, max_value) 
density = True            # If True, histogram shows probability density instead of counts
color = 'mediumorchid'     # Color of the bars
alpha = 0.7                # Transparency (0 = transparent, 1 = solid)
edgecolor = 'black'        # Color of the bar edges
label = 'STD Data distribution'      # Legend label
histtype = 'bar'           # Other options: 'step', 'stepfilled', 'barstacked'

x_range = np.linspace(min_value, max_value, 10000)

# Plotting the histogram
counts, bins_edges, patches = plt.hist(
    data, 
    bins=bins, 
    range=range_values, 
    density=density, 
    color=color, 
    alpha=alpha, 
    edgecolor=edgecolor, 
    label=label, 
    histtype=histtype
)

#Getting the Bin with max frecuency
max_index = np.argmax(counts)
max_bin_count = counts[max_index]

# fitting the histogram

bins_centers = 0.5 * (bins_edges[:-1] + bins_edges[1:])

pop, cov = curve_fit(gaussian, bins_centers, counts, p0=[max_bin_count, 4])

error=np.sqrt(np.diag(cov))

plt.plot(bins_centers, gaussian(bins_centers, *pop), 
         color='crimson', label=rf'$\mu$= {pop[0]:.2f}({error[0]:.2f}), $\sigma$= {pop[1]:.2f}({error[0]:.2f})',
         linewidth=4)

# Add labels, title, legend
plt.xlabel('Photon count', fontsize= 18)
plt.ylabel('Frecency', fontsize= 18)
plt.title(f'{dataset_title} STD Distribution', fontsize= 15)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.legend(fontsize=15)
plt.xlim(0,250)

# plt.savefig(f'{dataset_title} STD Histogram.jpg', format='jpg',
#             bbox_inches='tight')

plt.show()

#%%
# --------- VALUES OF STD OUTSIDE THE MAIN DISTRIBUTION --------------


sorted_std_per_channel=std_per_channel.copy()

sorted_std_per_channel.sort(reverse=True)


V1_std=sorted_std_per_channel[0] # Maximo valor
V2_std=sorted_std_per_channel[1] # Segundo Maximo valor

Index_1_std=std_per_channel.index(V1_std)
Index_2_std=std_per_channel.index(V2_std)

C1=channels[Index_1_std]
C2=channels[Index_2_std]

M1=mean_per_channel[Index_1_std]
M2=mean_per_channel[Index_2_std]

print('The value of ', round(V1_std,3), ' is on the ', C1, ' nm')
print('The value of ', round(V2_std,3), ' is on the ', C2, ' nm')


# -------------------------- PLOTTING -------------------------------

plt.figure(figsize=(12,8))
plt.errorbar(channels, mean_per_channel, fmt='.', 
             color='darkgoldenrod', yerr=std_per_channel,
             ecolor='mediumorchid', elinewidth=0.5, 
             capsize=5, capthick=0.5, 
            errorevery=1, alpha=0.9,
            label='Mean count')

# plt.scatter(C1, M1, marker='o', s=60, c='red', label='Point w/ Bigger STD {M1}')
plt.scatter(C2, M2, marker='o', s=60, c='red')

plt.xlabel('Wave Lenght (nm)', fontsize=20)
plt.ylabel('Intensity (A.U.)', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.title(f'{dataset_title} Mean Histogram', fontsize=20)

plt.legend(fontsize=18)

plt.savefig(f'{dataset_title} Mean Histogram.jpg', format='jpg',
            bbox_inches='tight')

plt.show()