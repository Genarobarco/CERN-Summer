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

def data_pablo(Concentration, Pressure, Tube_intensity, Voltage):

    path_datos = r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\Dataframe_Spectra_ArCF4"
    datos = pd.read_pickle(path_datos)

    mask1 = datos['Concentrations'] == Concentration 
    mask2 = datos['Pressure'] == Pressure
    mask3 = datos['Tube Intensity'] == Tube_intensity
    mask4 = datos['Voltages'] == Voltage
    fila = datos[mask1 & mask2 & mask3 & mask4]['Data']

    wave_pablo = fila.array[0][:,0]
    inte_pablo = fila.array[0][:,1]

    inte_pablo_norm = inte_pablo / max(inte_pablo)

    df_Pablo = pd.DataFrame({
        'Lambda': wave_pablo,
        'Counts': inte_pablo,
        'Counts_norm': inte_pablo_norm
        })
    
    return df_Pablo

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
      print(f"Carpeta creada en: {ruta_completa}")

    except Exception as e:
      print(f"Ocurrió un error al crear la carpeta: {e}")

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

def Excel_value(file_path, filters, target_column):

    df = pd.read_excel(file_path)
    
    mask = pd.Series(True, index=df.index)
    for col, val in filters.items():
        mask &= (df[col] == val)
    
    filtered_df = df[mask]
    
    if filtered_df.empty:
        print("No match found with the given filters.")
        return None
    
    return filtered_df[target_column].values[0]

def Sep_rut(Ruta, Mother_folder = 'Spectra_2025_Pablo_Raul_Genaro'):
  partes = Ruta.split('\\')

  for i, parte in enumerate(partes):

    if parte == Mother_folder:
      indicador = i

  Element = partes[indicador+1]
  Concentracion = int(partes[indicador+2])
  Presion = partes[indicador+3].split('_')[0]
  Volt_Amp = partes[indicador+4]

  if Presion.split('_'):
    Presion = float(Presion.replace('-', '.'))

  Ar_Concentration = 100 - int(Concentracion)

  return Element, Concentracion, Presion, Volt_Amp, Ar_Concentration

def Excel_writter(A, Ac, B, Bc, Pressure, VA, SV, SC, folder_path = '-'):

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
        'Folder Path': folder_path,
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

print('----------------------------------------------')
print('            DATA SORTING STARTED              ')
print('----------------------------------------------')
#------------- Ruta ----------------

Ruta=r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\N2\5\1_bar\40kV40mA\0V"
excel_path = r'C:\Users\genar\VSC code\CERN-Summer\Whole_Data.xlsx'

# ------- Analisis de Ruta -----------

folder_name = 'Analized_Data'
create_folder(Ruta, folder_name)

Element, Concentracion, Presion, Volt_Amp, Ar_Concentration = Sep_rut(Ruta)

# ----------- Values ---------------

Element_A = 'Ar'
Concentration_A = Ar_Concentration
Element_B = Element
Concentracion_B = Concentracion
Pressure = Presion

filters = {
    'Element A': Element_A,
    'Concentration A':  Concentration_A,
    'Element B': Element_B,
    'Concentration B': Concentracion_B,
    'Pressure (bar)': Pressure
}

print(' ')

if Excel_value(excel_path, filters, 'SC'):
   
  Saturation_current = Excel_value(excel_path, filters, 'SC')
  Saturation_volt = Excel_value(excel_path, filters, 'SV')

  print('The row already exists with a value of SV = ', Saturation_volt,'V and SC=', Saturation_current,'A')
  update = input('Do you wish to update? (Y/N) ')

  if update=='Y':
    Saturation_Voltage_UPD = float(input('UPDATED value of VOLTAGE: '))
    Saturation_Amp_UPD = float(input('UPDATED value of CURRENT: '))

    Excel_writter(Element_A, Concentration_A, 
                  Element_B, Concentracion_B, 
                  Pressure, '40kV40mA', SV = Saturation_Voltage_UPD, SC = Saturation_Amp_UPD)
    
    Saturation_current = Excel_value(excel_path, filters, 'SC')
    
  else:
      Saturation_current = Excel_value(excel_path, filters, 'SC')
     
else:

  Saturation_Voltage_NEW = float(input('NEW value of VOLTAGE: '))
  Saturation_Amp_NEW = float(input('NEW value of CURRENT: '))
  
  Excel_writter(Element_A, Concentration_A, 
              Element_B, Concentracion_B, 
              Pressure, '40kV40mA', SV = Saturation_Voltage_NEW, SC = Saturation_Amp_NEW)
  
  Saturation_current = Excel_value(excel_path, filters, 'SC')

print(' ')

Ruta_BG=glob(f'{Ruta}\DataBG\*.txt')

name = f'Ar{Element}_{Ar_Concentration}{Concentracion}_{Presion}'

# -------- BG behaviour --------

BG_Tender(Ruta_BG, f'{Ruta}\{folder_name}', label_tender = f'{name}', p0=[4200,1000,1], 
              bins_histo=1000, distribution = landau,
              Label_histo = 'Counts_BG', lim = True, 
              x_min= 2000, x_max=12000, plot_histos= False,
              color_tender = 'crimson')

# ----- Mean average for all files ---------

df = Mean_BG(Ruta_BG)

Lambda = df['Lambda']
Mean_Counts = df['Mean_Counts']
Std_Counts = df['Std_Counts']

plt.figure(figsize=(12,8))
plt.plot(Lambda, Std_Counts, label=f'{name}-STDperChannel', color='blue',
             marker='o', markersize=1)
plt.xlabel('Wavelength (nm)', fontsize=15)
plt.ylabel('Photon Count (A.U.)', fontsize=15)
plt.legend(loc='upper right', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=18)

plt.savefig(f'{Ruta}\{folder_name}\{name}_STDMeanChannel.jpg', format='jpg', bbox_inches='tight')

plt.figure(figsize=(12,8))
plt.errorbar(Lambda, Mean_Counts, yerr=Std_Counts,
             label=f'{name}-Mean BGSpectrum', color='black',
             fmt=':.', markersize=1)
plt.xlabel('Wavelength (nm)', fontsize=15)
plt.ylabel('Photon Count (A.U.)', fontsize=15)
plt.legend(loc='upper right', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.savefig(f'{Ruta}\{folder_name}\{name}_MHisto.jpg', format='jpg', bbox_inches='tight')

Histo(Std_Counts, landau, [60,15,10], 1200, f'Std {name}', 
      lim=True, x_min=0, x_max=300, save_name=f'{Ruta}\{folder_name}\{name}_Std')

Histo(Mean_Counts, landau, [4200,1000,1], 800, f'Mean {name}',
      color='Navy', lim=True, x_min=2000, x_max=14000, 
      save_name=f'{Ruta}\{folder_name}\{name}_Mean')

# ------- Calibrated Spectrum ----------

filtered_data = RP(Ruta)

df_bgSpec = filtered_data['bgSpectrum']             # raw Background
df_corrSpec = filtered_data['correctedSpectrum']
df_raw = filtered_data['rawSpectrum']               # raw Signal

df_results = filtered_data['calibratedResults']

plt.figure(figsize=(12,8))

plt.plot(df_results['Lambda'], df_results['Counts'], 
         label=f'{name}', color='crimson', linewidth = 0.5, alpha = 1)
plt.legend(fontsize=20)
plt.xlabel('Wavelenght', fontsize=15)
plt.ylabel('Absolute Counts (A.U.)', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.grid()

plt.savefig(f'{Ruta}\{folder_name}\{name}_CalibratedResults.jpg', format='jpg', 
            bbox_inches='tight', dpi = 300)

plt.figure(figsize=(12,8))

plt.plot(df_results['Lambda'], df_results['Counts_norm'], 
         label=f'{name}', color='crimson', linewidth = 0.5, alpha = 1)
plt.legend(fontsize=20)
plt.xlabel('Wavelenght', fontsize=15)
plt.ylabel('Normalize Counts (A.U.)', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.grid()

plt.savefig(f'{Ruta}\{folder_name}\{name}_CalibratedResults_norm.jpg', format='jpg', 
            bbox_inches='tight', dpi = 300)


# ---------------- Photon por Electron ----------------

print(' ')
print('The Saturation Current is: ', Saturation_current, 'A')
print()

N_e = Saturation_current / (-1.602176634e-19)

intensity_Results_photon = df_results['Counts']/N_e

plt.figure(figsize=(12,8))

plt.plot(df_results['Lambda'], intensity_Results_photon, 
         label=f'{Saturation_current} A', color='blue', linewidth = 0.5)

plt.legend(fontsize=20)
plt.xlabel('Wavelenght', fontsize=15)
plt.ylabel(r'Photons per e$^-$ (A.U.)', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.grid()

plt.savefig(f'{Ruta}\{folder_name}\{name}_Photon.jpg', format='jpg', 
            bbox_inches='tight', dpi = 300) 

plt.show(block=False)
plt.pause(0.1)
input("Press enter to close all figures...")
plt.close('all')