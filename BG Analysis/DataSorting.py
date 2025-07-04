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

def gaussian(x, mu, sigma, A):
    return (A / (sigma * np.sqrt(2 * np.pi))) *np.exp(- (x - mu)**2 
                                                       / (2 * sigma**2))

def landau(x, mpv, eta, A):
    xi = (x - mpv) / eta
    return A * np.exp(-0.5 * (xi + np.exp(-xi)))

def Reader(Rutas):

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

  mean_per_channel=[]
  std_per_channel=[]

  for j in np.arange(0,2048):

    lista=[]

    for i in np.arange(0,len(Count_List)):

      lista.append(Count_List[i][j])
      
    mean_per_channel.append(np.mean(lista))
    std_per_channel.append(np.std(lista))

  channels=Lambda_List[0]

  return mean_per_channel, std_per_channel, channels

def Histo(list, distribution, p0, bins_value, Label_value, density = True,
    color = 'mediumorchid', color_plot = 'crimson', alpha = 0.7, edgecolor = 'black',      # Legend label
    histtype = 'bar',lim = False, x_min = 0, x_max = 1000):

    min_value=min(list)
    max_value=max(list)

    plt.figure(figsize=(12,8))

    # Histogram configuration
    data= list
    bins = bins_value
    range_values = (min_value, max_value)
    label = Label_value

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

    # fitting the histogram

    bins_centers = 0.5 * (bins_edges[:-1] + bins_edges[1:])

    pop, cov = curve_fit(distribution, bins_centers, counts, p0=p0)

    error=np.sqrt(np.diag(cov))

    plt.plot(x_range, distribution(x_range, *pop), 
            color= color_plot, label=f'{pop}',
            linewidth=4)

    # Add labels, title, legend
    plt.xlabel('Photon count', fontsize= 18)
    plt.ylabel('Frecency', fontsize= 18)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.legend(fontsize=15)
    
    if lim:
        plt.xlim(x_min,x_max)

    plt.show()

Rutas=glob(r'C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\CF4\1\1_bar\0V\test_pure\DataBG\*.txt')

mean_per_channel = Reader(Rutas)[0]
std_per_channel = Reader(Rutas)[1]

Histo(mean_per_channel, landau, [4000,2000,1], 200, 
      'Counts BG', color = 'blue', lim = True, 
      x_min= 3000, x_max=10000)

Histo(std_per_channel, gaussian, [50,20,1], 1000, 'STD BG', 
      color = 'orange', lim = True, x_min= -10, x_max=120)