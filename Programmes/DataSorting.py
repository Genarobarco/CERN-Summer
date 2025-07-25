#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math as math
from glob import glob
from Functions import Sep_rut, BG_Tender, Mean_BG, Histo, RP, Excel_value, Excel_writter, landau, create_folder


print(' ')
print('----------------------------------------------')
print('            DATA SORTING STARTED              ')
print('----------------------------------------------')
print(' ')

#------------- Ruta ----------------

Ruta=r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\N2\0-1\3-5_bar\40kV40mA\0V"
Ruta_Candela = r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\CF4\5\5_bar\40kV40mA\After_WindowChange"

# ----------- Reference --------------

Candela_FD = RP(Ruta_Candela)
df_candela = Candela_FD['calibratedResults']

filters_candela = {
    'Element A': 'Ar',
    'Concentration A':  95,
    'Element B': 'CF4',
    'Concentration B': 5,
    'Pressure (bar)': 5
}

Current_candela = Excel_value(filters_candela, 'SC')
NumElectronsCandela = Current_candela / (-1.602176634e-19)

Candela_phe = df_candela['Counts'] / NumElectronsCandela

# ------- Analisis de Ruta -----------

folder_name = 'Analized_Data'
ruta_data = create_folder(Ruta, folder_name)

Element, Concentracion, Presion, Volt_Amp, Ar_Concentration = Sep_rut(Ruta)

filters = {
    'Element A': 'Ar',
    'Concentration A':  Ar_Concentration,
    'Element B': Element,
    'Concentration B': Concentracion,
    'Pressure (bar)': Presion
}

print(' ')

if Excel_value(filters, 'SC'):
 
  Saturation_current = Excel_value(filters, 'SC')
  Saturation_volt = Excel_value(filters, 'SV')
  SC_3kV = Excel_value(filters, 'C3kV')

  print('The row already exists with a value of SV = ', Saturation_volt,'V and SC=', Saturation_current,'A and', SC_3kV, 'A at 3kV' )
  update = input('Do you wish to update? (Y/N) ')

  if update=='Y':
    Saturation_Voltage_UPD = float(input('UPDATED value of VOLTAGE: '))
    Saturation_Amp_UPD = float(input('UPDATED value of CURRENT: '))
    SC_3kV_UPD = float(input('UPDATED value of CURRENT AT 3kV: '))

    Excel_writter('Ar', Ar_Concentration, 
                  Element, Concentracion, 
                  Presion, '40kV40mA', 
                  SV = Saturation_Voltage_UPD, 
                  SC = Saturation_Amp_UPD,
                  Current_3kV = SC_3kV_UPD)
    
    Saturation_current = Excel_value(filters, 'SC')
    Saturation_current_3kV = Excel_value(filters, 'C3kV')
    
  else:
      Saturation_current = Excel_value(filters, 'SC')
      Saturation_current_3kV = Excel_value(filters, 'C3kV')
     
else:

  Saturation_Voltage_NEW = float(input('NEW value of VOLTAGE: '))
  Saturation_Amp_NEW = float(input('NEW value of CURRENT: '))
  SC_3kV_NEW = float(input('NEW value of CURRENT AT 3kV: '))
  
  Excel_writter('Ar', Ar_Concentration, 
              Element, Concentracion, 
              Presion, '40kV40mA', SV = Saturation_Voltage_NEW, SC = Saturation_Amp_NEW,
              Current_3kV=SC_3kV_NEW)
  
  Saturation_current = Excel_value(filters, 'SC')
  Saturation_current_3kV = Excel_value(filters, 'C3kV')

if Excel_value(filters, 'C3kV')!=0:
  err_SC = Excel_value(filters, 'Err SC')

else:
  print('Saturation Current at 3kV is not written.')
  error_input = str(input('Do you wish to USE STANDARD (US) or WRITTE (W) it? '))

  if error_input=='US':
    err_SC = 0.05e-7
    print('Using Standar Current Error of:', err_SC, 'A')

  else:
    SC_3kV_Write = float(input('NEW value of CURRENT AT 3kV: '))
    Sv = Excel_value(filters, 'SV')
    Sc = Excel_value(filters, 'SC')

    Excel_writter('Ar', Ar_Concentration, 
                Element, Concentracion, 
                Presion, '40kV40mA', SV=Sv, SC=Sc,
                Current_3kV=SC_3kV_Write)
    
    err_SC = Excel_value(filters, 'Err SC')

print(' ')

Ruta_BG=glob(f'{Ruta}\DataBG\*.txt')

name = f'Ar{Element}_{Ar_Concentration}{Concentracion}_{Presion}'

# ------------- Analysis ------------------

filtered_data = RP(Ruta)

df_results = filtered_data['calibratedResults']
df_bgSpec = filtered_data['bgSpectrum']             # raw Background
df_corrSpec = filtered_data['correctedSpectrum']
df_raw = filtered_data['rawSpectrum']               # raw Signal

N_e = Saturation_current / (-1.602176634e-19)
err_NumeroElectrones = err_SC/ (-1.602176634e-19)

print(' ')
print('The Saturation Current is: ', Saturation_current, 'A')
print('With an error of: ', err_SC, 'A')
print()

df_phe = df_results['Counts'].clip(lower = 0)/N_e
err_phe = np.sqrt((df_results['Err_Counts']/N_e)**2+(df_results['Counts']*err_NumeroElectrones/(N_e)**2)**2)

df = pd.DataFrame({
          'Lambda': df_results['Lambda'],
          'Counts': df_results['Counts'],
          'Err_Counts': np.sqrt(df_results['Counts'].clip(lower=0)),
          'Counts_norm': df_results['Counts']/max(df_results['Counts']),
          'Err_Counts_norm': np.sqrt(df_results['Counts'].clip(lower=0))/max(df_results['Counts']),
          'Phe':df_phe,
          'Err_Phe': err_phe
          })

df.to_csv(f'{ruta_data}\{name}_AllData.txt', sep='\t', index=False)

print(f'All data saved in: {ruta_data}')

print(' ')

plot_input = str(input('Do you wish to plot? (Y/N) '))

# ------------------------------------- PLOTS ---------------------------------------------
# -------- BG behaviour --------

BG_Tender(Ruta_BG, f'{ruta_data}', label_tender = f'{name}', p0=[4200,1000,1], 
              bins_histo=1000, distribution = landau,
              Label_histo = 'Counts_BG', lim = True, 
              x_min= 2000, x_max=12000, plot_histos= False,
              color_tender = 'crimson')

# ----- Mean average for all files ---------

df = Mean_BG(Ruta_BG)

Lambda = df['Lambda']
Mean_Counts = df['Mean_Counts']
Std_Counts = df['Std_Counts']

plt.figure(figsize=(16,9))
plt.plot(Lambda, Std_Counts, label=f'{name}-STDperChannel', color='blue',
             marker='o', markersize=1)
plt.xlabel('Wavelength (nm)', fontsize=15)
plt.ylabel('Photon Count (A.U.)', fontsize=15)
plt.legend(loc='upper right', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=18)

plt.savefig(f'{ruta_data}\{name}_STDMeanChannel.jpg', format='jpg', bbox_inches='tight')

plt.figure(figsize=(16,9))
plt.errorbar(Lambda, Mean_Counts, yerr=Std_Counts,
             label=f'{name}-Mean BGSpectrum', color='black',
             fmt=':.', markersize=1)
plt.xlabel('Wavelength (nm)', fontsize=15)
plt.ylabel('Photon Count (A.U.)', fontsize=15)
plt.legend(loc='upper right', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.savefig(f'{ruta_data}\{name}_MHisto.jpg', format='jpg', bbox_inches='tight')

Histo(Std_Counts, landau, [60,15,10], 1200, f'Std {name}', 
      lim=True, x_min=0, x_max=300, save_name=f'{ruta_data}\{name}_Std')

Histo(Mean_Counts, landau, [4200,1000,1], 800, f'Mean {name}',
      color='Navy', lim=True, x_min=2000, x_max=14000, 
      save_name=f'{ruta_data}\{name}_Mean')

# ------- Calibrated Spectrum ----------

plt.figure(figsize=(16,9))
plt.plot(df_results['Lambda'], df_results['Counts'], 
         label=f'{name}', color='crimson', linewidth = 0.5, alpha = 1)

plt.plot(df_candela['Lambda'], df_candela['Counts'], 
         label=f'Reference', color='darkviolet', linewidth = 0.5, alpha = 1)

plt.legend(fontsize=20)
plt.xlabel('Wavelenght', fontsize=15)
plt.ylabel('Absolute Counts (A.U.)', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.grid()

plt.savefig(f'{ruta_data}\{name}_CalibratedResults.jpg', format='jpg', 
            bbox_inches='tight', dpi = 300)


plt.figure(figsize=(16,9))
plt.plot(df_results['Lambda'], df_results['Counts_norm'], 
         label=f'{name}', color='crimson', linewidth = 0.5, alpha = 1)
plt.plot(df_candela['Lambda'], df_candela['Counts_norm'], 
         label=f'Reference', color='darkviolet', linewidth = 0.5, alpha = 1)
plt.legend(fontsize=20)
plt.xlabel('Wavelenght', fontsize=15)
plt.ylabel('Normalize Counts (A.U.)', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.grid()

plt.savefig(f'{ruta_data}\{name}_CalibratedResults_norm.jpg', format='jpg', 
            bbox_inches='tight', dpi = 300)


# ---------------- Photon por Electron ----------------


plt.figure(figsize=(16,9))

plt.plot(df_results['Lambda'], df_phe,  
         label=f'Ar/{Element} {Ar_Concentration}/{Concentracion} {Presion}bar ({Saturation_current}A)', color='blue', linewidth = 0.5)

plt.plot(df_candela['Lambda'], Candela_phe, 
         label=f'Ar/CF4 95/5 5bar ({Current_candela}A)', color='darkviolet', linewidth = 0.5)

plt.legend(fontsize=20)
plt.xlabel('Wavelenght', fontsize=15)
plt.ylabel(r'$\gamma$ / e$^-$ (A.U.)', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.grid()

plt.savefig(f'{ruta_data}\{name}_Photon.jpg', format='jpg', 
            bbox_inches='tight', dpi = 300) 


if plot_input == 'Y':
  plt.show(block=False)
  plt.pause(0.1)
  input("Press enter to close all figures...")
  plt.close('all')

else:
  plt.close('all')