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

def Mean_BG(Rutas):

  # Lo que hace este codigo es tomar una ruta, la misma en teoria posee
  # varios archivos de BG. Al final toma el valor medio por canal y la
  # desviacion estandar por canal.

  Count_List=[]
  Lambda_List=[]
  Indice= 0

  for ruta in Rutas:

    array= pd.read_csv(ruta, names=['Lambda', 'Counts'], sep='\s+', 
                      header=None, engine='python')
    
    Count_List.append(array['Counts'])
    Lambda_List.append(array['Lambda'])

    Indice += 1

  mean_per_channel=[]
  std_per_channel=[]

  # Convertimos la lista de cuentas en un array 2D (archivos x canales)
  counts_matrix = np.array(Count_List)

  # Calculamos media y desviación por canal (eje 0 = entre archivos)
  mean_per_channel = np.mean(counts_matrix, axis=0)
  std_per_channel = np.std(counts_matrix, axis=0)

  channels=Lambda_List[0]

  df = pd.DataFrame({
    'Lambda': channels,
    'Mean_Counts': mean_per_channel,
    'Std_Counts': std_per_channel
  })

  return df

def Histo(list, distribution, p0, bins_value, Label_value, density = True,
    color = 'mediumorchid', color_plot = 'crimson', alpha = 0.7, edgecolor = 'black',      # Legend label
    histtype = 'bar',lim = False, x_min = 0, x_max = 1000, plot = True, 
    save_name='Poner_nombre_Histos'):

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
  plt.title(save_name)
  plt.legend(fontsize=15)
  
  if lim:
      plt.xlim(x_min,x_max)

  if plot:

    plt.savefig(f'{save_name}.jpg', format='jpg', bbox_inches='tight')
    plt.show()

  else:
    plt.close()

  return pop, cov

def BG_Tender(Rutas, label_tender, p0=[6000,100,1], 
              bins_histo=500, distribution = landau,
              Label_histo = 'Counts_BG', lim = False, 
              x_min= 2000, x_max=14000, plot_histos= False,
              color_tender = 'crimson'):
  
  counts_values = []

  for r in Rutas:

    df = Mean_BG([r])

    value_per_channel = df['Mean_Counts'] # Aca dice mean pero en verdad es solo 1 dato el que esta contando
                                        # ya que solo entra 1 archivo por interacion

    counts = Histo(value_per_channel, distribution, p0=p0, bins_value=bins_histo, 
          Label_value=Label_histo, color = 'blue', lim = lim, 
          x_min= x_min, x_max=x_max, plot= plot_histos)[0][0]

    counts_values.append(counts)

  max_mean = max(counts_values)
  min_mean = min(counts_values)

  max_index = counts_values.index(max_mean)
  min_index = counts_values.index(min_mean)

  percent = ((max_mean - min_mean)/min_mean)*100

  print('------ BG Behaviour -----')
  print('Max counts = ', round(max_mean,2))
  print('Min counts = ', round(min_mean,2))
  print('Variation: ', round(percent,2),'%')
  print('-------------------------')

  x_length = range(len(counts_values))

  plt.figure(figsize=(12,8))
  plt.errorbar(x_length, counts_values, 
              marker='v', linestyle=':', color=color_tender, 
              markersize = 10, label=label_tender)
  
  plt.vlines(x=max_index, ymin= min_mean, ymax=max_mean, label=rf'{percent:.2f}%')

  plt.legend(fontsize=20)
  plt.xlabel('File Number', fontsize=20)
  plt.ylabel('BG mean counts', fontsize=20)
  plt.tick_params(axis='both', labelsize=20)

  plt.savefig(f'{label_tender}_BGMeanBehaviour.jpg', format='jpg', bbox_inches='tight' )

  plt.show()

def RP(base_path):
  """
  Returns the full path to the first file in the 'Results' subfolder 
  of `base_path` that ends with '-calibratedResults.csv'.

  Parameters:
      base_path (str): Path to the folder containing the 'Results' subfolder.

  Returns:
      str: Full path to the matching file, or None if not found.
  """
  results_path = os.path.join(base_path, 'Results')
  matched_files = glob(os.path.join(results_path, '*-calibratedResults.csv'))

  if matched_files:
      return matched_files[0]
  else:
      return None
    
#%%
# -------- Path ---------

Mother_path =r'C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro'
Parameters=r'\N2\100\1_bar\40kV40mA'
Folder_name='30s_gena'

Ruta=f'{Mother_path}\{Parameters}\{Folder_name}'
Ruta_BG=glob(f'{Ruta}\DataBG\*.txt')
Ruta_Results = RP(Ruta)

name = 'N2_100_1bar_30s'
#%%

# ----- Mean average for all files ---------
df = Mean_BG(Ruta_BG)

Lambda = df['Lambda']
Mean_Counts = df['Mean_Counts']
Std_Counts = df['Std_Counts']

plt.figure(figsize=(12,8))
plt.plot(Lambda, Mean_Counts, label=f'Mean BGSpectrum - {name}',
         color='darkviolet')
plt.xlabel('Wavelength (nm)', fontsize=15)
plt.ylabel('Photon Count (A.U.)', fontsize=15)
plt.legend(loc='upper right', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.savefig(f'MHist_{name}.jpg', format='jpg', bbox_inches='tight')
plt.show()

Histo(Std_Counts, gaussian, [60,15,10], 1000, 'Mean Counts all BG', 
      lim=True, x_min=0, x_max=150, save_name=f'Std_{name}')

Histo(Mean_Counts, landau, [8000,4000,10], 100, 'Mean Counts all BG',
      color='Navy', lim=True, x_min=2000, x_max=20000, 
      save_name=f'Mean_{name}')


#%%
# -------- BG behaviour --------

BG_Tender(Ruta_BG, label_tender = 'N2_100_1b_30s', p0=[6000,100,1], 
              bins_histo=500, distribution = landau,
              Label_histo = 'Counts_BG', lim = True, 
              x_min= 2000, x_max=14000, plot_histos= False,
              color_tender = 'crimson')


# ------- Calibrated Spectrum ----------

df_1 = pd.read_csv(Ruta_Results)

wavelength_gena = df_1['wavelength']
intensity_gena = df_1['intensity']

intensity_norm_gena = intensity_gena

plt.plot(wavelength_gena, intensity_norm_gena / max(intensity_norm_gena) , 
         label='try2 - 5/30 Señal - 10/30 BG', color='red')
plt.legend()
plt.xlabel('Wavelenght')
plt.ylabel('Normalize Counts (A.U.)')

# plt.savefig('try3_N2_100_1bar_5-30_10-30.jpg', format='jpg', bbox_inches='tight' )

plt.show()