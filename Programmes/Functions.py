#%%
import os
import sys
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math as math
from glob import glob
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.special import factorial

excel_path = r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\Whole_Data.xlsx"

def extraer_presion(ruta):
    """Extrae la presión en float desde la ruta."""
    partes = ruta.split(os.sep)
    for parte in partes:
        if '_bar' in parte:
            presion_str = parte.replace('_bar', '').replace('-', '.')
            try:
                return float(presion_str)
            except ValueError:
                return float('inf')  # Por si hay algún error, lo manda al final
    return float('inf')

def integral(dt, col_x, err_x, col_y, err_y, lim_inf, lim_sup):
    mask = (dt[col_x] >= lim_inf) & (dt[col_x] <= lim_sup)
    dt_filtered = dt[mask].sort_values(by=col_x).reset_index(drop=True)

    x = dt_filtered[col_x].values
    y = dt_filtered[col_y].values
    dy = dt_filtered[err_y].values  # Uncertainty in y
    dx = dt_filtered[err_x].values if err_x in dt_filtered.columns else np.diff(x, prepend=x[0])

    # Trapezoidal integration
    area = np.trapezoid(y=y, x=x)

    error_squared = 0
    for i in range(len(x)-1):
        base = x[i+1] - x[i]
        avg_y_err_sq = base**2 * (dy[i+1]**2+dy[i]**2)
        error_squared += avg_y_err_sq

    total_error = np.sqrt(error_squared)

    return area, total_error

def data_pablo(Concentration, Pressure, Tube_intensity, Voltage):

    path_datos = r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\Dataframe_Spectra_ArCF4"
    datos = pd.read_pickle(path_datos)

    mask1 = datos['Concentrations'] == Concentration 
    mask2 = datos['Pressure'] == Pressure
    mask3 = datos['Tube Intensity'] == Tube_intensity
    mask4 = datos['Voltages'] == Voltage
    fila = datos[mask1 & mask2 & mask3 & mask4]['Data']
    corriente_pablo = datos[mask1 & mask2 & mask3 & mask4]['Currents']

    wave_pablo = fila.array[0][:,0]
    inte_pablo = fila.array[0][:,1]

    inte_pablo_norm = inte_pablo / max(inte_pablo)

    df_Pablo = pd.DataFrame({
        'Lambda': wave_pablo,
        'Counts': inte_pablo,
        'Counts_norm': inte_pablo_norm
        })
    
    return df_Pablo, corriente_pablo

def create_folder(ruta_base, nombre_carpeta):
  ruta_completa = os.path.join(ruta_base, nombre_carpeta)
  
  if os.path.exists(ruta_completa):
    print(f"The folder already exists: {ruta_completa}")
    overwrite = str(input('Do you wish to overwrite? (Y/N) '))

    if overwrite == 'Y':

      shutil.rmtree(ruta_completa)

      try:
        os.makedirs(ruta_completa)
        print(f"created folder on: {ruta_completa}")

      except Exception as e:
        print(f"An error has happend on: {e}")
    
    else:
        other = str(input('Do you want to create another folder? (Y/N) '))

        if other == 'Y':
           
           new_name = input('insert new name: ')
           ruta_completa = os.path.join(ruta_base, new_name)
           os.makedirs(ruta_completa)
           print(f"Folder created on: {ruta_completa}")

        else:
          input('Programme stoped')
          sys.exit()

  else:
    try:
      os.makedirs(ruta_completa)
      print(f"Folder created on: {ruta_completa}")

    except Exception as e:
      print(f"Ocurrió un error al crear la carpeta: {e}")

  return ruta_completa

def gaussian(x, mu, sigma, A):
    return (A / (sigma * np.sqrt(2 * np.pi))) *np.exp(- (x - mu)**2 
                                                       / (2 * sigma**2))

def landau(x, mpv, eta, A):
    xi = (x - mpv) / eta
    return A * np.exp(-0.5 * (xi + np.exp(-xi)))

def poisson(x, lamb, A):
    return A * (lamb**x) * np.exp(-lamb) / factorial(x)

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
    save_name='Ajuste_Histos'):

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

  if plot:

    plt.savefig(f'{save_name}.jpg', format='jpg', bbox_inches='tight')
    # plt.show()

  else:
    plt.close()

  return pop, cov

def BG_Tender(Rutas, save_rute, label_tender, p0=[6000,100,1], 
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

  plt.savefig(f'{save_rute}\{label_tender}_BGMeanBehaviour.jpg', format='jpg', bbox_inches='tight' )

def RP(base_path):
  
  results_path = os.path.join(base_path, 'Results')
  files = ['calibratedResults','bgSpectrum', 'correctedSpectrum', 'rawSpectrum']

  data = {}

  for i in files:

    matched_files = glob(os.path.join(results_path, f'*-{i}.csv'))

    if matched_files:
        df_rute = pd.read_csv(matched_files[0], sep = ',')
        df = pd.DataFrame({
            'Lambda': df_rute['wavelength'],
            'Counts': df_rute['intensity'],
            'Counts_norm': df_rute['intensity']/max(df_rute['intensity']),
            'Err_Counts': np.sqrt(df_rute['intensity'].clip(lower=0)),
            })
        data[i] = df
    else:
        print(f'{i} file did not found')

  return data

def Sep_rut(Ruta, Mother_folder = 'Spectra_2025_Pablo_Raul_Genaro'):
  partes = Ruta.split('\\')

  for i, parte in enumerate(partes):

    if parte == Mother_folder:
      indicador = i

  Element = partes[indicador+1]

  if '-' in partes[indicador+2]:
    Concentracion = float(partes[indicador+2].replace('-', '.'))

  else:
     Concentracion = int(partes[indicador+2])

  Presion = partes[indicador+3].split('_')[0]
  Volt_Amp = partes[indicador+4]

  if '-' in Presion:
    Presion = float(Presion.replace('-', '.'))

  else:
    Presion = int(Presion)

  Ar_Concentration = 100 - Concentracion

  return Element, Concentracion, Presion, Volt_Amp, Ar_Concentration

def Excel_writter(A, Ac, B, Bc, Pressure, VA, SV, SC, Current_3kV):

  df = pd.read_excel(excel_path)

  # New values
  new_data = {
      'Element A': A,
      'Concentration A': Ac,
      'Element B': B,
      'Concentration B': Bc,
      'Pressure (bar)': Pressure,
      'Volt-Amp': VA,
      'SV': SV,
      'SC': SC,
      'C3kV': Current_3kV,
  }

  # Filter to check if a row with the same Element B, Concentration B, and Pressure exists
  mask = (
      (df['Element B'] == new_data['Element B']) &
      (df['Concentration B'] == new_data['Concentration B']) &
      (df['Pressure (bar)'] == new_data['Pressure (bar)'])
  )

  if df[mask].empty:
      # No match found, append new row
      df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
      print("New row added.")
  else:
      # Match found, update that row
      for key, value in new_data.items():
          df.loc[mask, key] = value
      print("Row updated.")

  # Save back to Excel
  df.to_excel(excel_path, index=False)

def Excel_value(filters, target_column):

  df = pd.read_excel(excel_path)
  
  mask = pd.Series(True, index=df.index)
  for col, val in filters.items():
      mask &= (df[col] == val)
  
  filtered_df = df[mask]
  
  if filtered_df.empty:
      print("No match found with the given filters.")
      return None
  
  return filtered_df[target_column].values[0]

def Excel_value_WP(Path, filters, target_column):

  df = pd.read_excel(Path)
  
  mask = pd.Series(True, index=df.index)
  for col, val in filters.items():
      mask &= (df[col] == val)
  
  filtered_df = df[mask]
  
  if filtered_df.empty:
      print("No match found with the given filters.")
      return None
  
  return filtered_df[target_column].values[0]

def wavelength_to_rgb(wavelength):
    gamma = 0.8
    intensity_max = 255

    if 380 <= wavelength <= 440:
        R = -(wavelength - 440) / (440 - 380)
        G = 0.0
        B = 1.0
    elif 440 <= wavelength <= 490:
        R = 0.0
        G = (wavelength - 440) / (490 - 440)
        B = 1.0
    elif 490 <= wavelength <= 510:
        R = 0.0
        G = 1.0
        B = -(wavelength - 510) / (510 - 490)
    elif 510 <= wavelength <= 580:
        R = (wavelength - 510) / (580 - 510)
        G = 1.0
        B = 0.0
    elif 580 <= wavelength <= 645:
        R = 1.0
        G = -(wavelength - 645) / (645 - 580)
        B = 0.0
    elif 645 <= wavelength <= 750:
        R = 1.0
        G = 0.0
        B = 0.0
    else:
        R = G = B = 0.0

    if 380 <= wavelength <= 420:
        factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
    elif 420 <= wavelength <= 700:
        factor = 1.0
    elif 700 <= wavelength <= 750:
        factor = 0.3 + 0.7 * (750 - wavelength) / (750 - 700)
    else:
        factor = 0.0

    R = int(round(intensity_max * (R * factor)**gamma))
    G = int(round(intensity_max * (G * factor)**gamma))
    B = int(round(intensity_max * (B * factor)**gamma))

    return (R/255.0, G/255.0, B/255.0)
