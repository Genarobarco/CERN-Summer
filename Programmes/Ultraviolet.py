#%%
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import math as math
from glob import glob
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.special import factorial

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

# Errores:
err_SC = 0.05e-7 #A
err_lambda = 50 #nm
err_pressure = 5e-3 #bar
#----------------

element_mix = 'N2' 
Mix_Integrals = {}
list_concentraciones = [0.1,1,5, 10]

data_integral=[]

integration_limits = [200, 800]

excel_path = r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\Whole_Data.xlsx"
base_path= rf'C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\{element_mix}'

for Concentracion_N2 in list_concentraciones:

    if '.' in str(Concentracion_N2):
        Concentration_mix = str(Concentracion_N2).replace('.','-')

    else:
        Concentration_mix = Concentracion_N2


    path = f'{base_path}\{Concentration_mix}'
    pattern = os.path.join(path, '*_bar', '40kV40mA', '0V')

    rutas = glob(pattern)
    rutas_ordenadas = sorted(rutas, key=extraer_presion)

    pressures = []
    currents = []
    voltages = []
    electrons = []
    data = []

    for i in rutas_ordenadas:

        Element, Concentracion, Presion, Volt_Amp, Ar_Concentration = Sep_rut(i)

        filters = {
            'Element A': 'Ar',
            'Concentration A':  Ar_Concentration,
            'Element B': Element,
            'Concentration B': Concentracion,
            'Pressure (bar)': Presion
        }

        Saturation_current = Excel_value(excel_path, filters, 'SC')
        Saturation_volt = Excel_value(excel_path, filters, 'SV')

        filtered_data = RP(i)
        df_results = filtered_data['calibratedResults']

        N_e = Saturation_current / (-1.602176634e-19)
        err_NumeroElectrones = err_SC/ (-1.602176634e-19)

        df_phe = df_results['Counts']/N_e
        err_phe = np.sqrt((df_results['Err_Counts']/N_e)**2+(df_results['Counts']*err_NumeroElectrones/(N_e)**2)**2)

        pressures.append(Presion)
        currents.append(Saturation_current)
        voltages.append(Saturation_volt)
        electrons.append(N_e)

        df = pd.DataFrame({
                'Lambda': df_results['Lambda'],
                'Counts': df_results['Counts'],
                'Err_Counts': np.sqrt(df_results['Counts'].clip(lower=0)),
                'Counts_norm': df_results['Counts']/max(df_results['Counts']),
                'Err_Counts_norm': np.sqrt(df_results['Counts'].clip(lower=0))/max(df_results['Counts']),
                'Phe':df_phe,
                'Err_Phe': err_phe
                })

        data.append(df)

#   ---- Calculo Integral ----

    print('Lista de presiones: ', pressures)
    integrales = []
    err_integral = []
    for ValPressure in pressures:
        sum, err = integral(data[int(ValPressure)], 
                       'Lambda', err_lambda,
                       'Phe', 'Err_Phe',
                       *integration_limits)
        integrales.append(sum)
        err_integral.append(err)

    df = pd.DataFrame({
            'Pressures': pressures,
            'integrals': integrales, 
            'Err_int': err_integral
            })

    Mix_Integrals[Concentracion_N2] = df
    data_integral.append(df)

#   --------------------------

    if integrales[len(integrales)-1]>integrales[0]:
        rango = sorted(range(len(pressures)), reverse= True)
    else:
        rango = sorted(range(len(pressures)), reverse= False)

    norm = mcolors.Normalize(vmin=min(pressures), vmax=max(pressures))
    colormap = plt.colormaps['turbo']

    fig, ax = plt.subplots(figsize=(16, 9))

    for i in rango:

        mask = (data[i]['Lambda'] >= integration_limits[0]) & (data[i]['Lambda'] <= integration_limits[1])
        data_filtered = data[i][mask].sort_values(by='Lambda')

        color = colormap(norm(pressures[i]))
        ax.errorbar(data_filtered['Lambda'], 
                    data_filtered['Phe'], 
                    yerr=data_filtered['Err_Phe'],
                    label=f'{pressures[i]} bar',
                    color=color,
                    linewidth=0.5)
        
        ax.fill_between(data_filtered['Lambda'], 0, data_filtered['Phe'],
                        color=color, alpha = 0.7)

    ax.set_xlabel(r'\lambda (nm)', fontsize=15)
    ax.set_ylabel(r'$\gamma$ / e$^-$ (A.U)', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.grid()
    ax.legend(fontsize=13, title='Pressures', title_fontsize=12)

    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Pressure (bar)', fontsize=13)
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()

    plt.title(f'Ar/{Element} {100-Concentracion_N2}/{Concentracion_N2}')

    plt.savefig(f'Ar{Element}_{Ar_Concentration}{Concentracion}_Ultraviolet_phe.jpg', format='jpg', 
                bbox_inches='tight', dpi = 300) 

range_concentraciones = np.linspace(list_concentraciones[0],
                                    list_concentraciones[len(list_concentraciones)-1],
                                    10)
norm = mcolors.Normalize(vmin=min(range_concentraciones), vmax=max(range_concentraciones))
colormap = plt.colormaps['turbo']
fmt_list = [':^', ':v', ':o', ':s']

fig, ax = plt.subplots(figsize=(16, 9))

indicador = 0
for i in list_concentraciones:

    color = colormap(norm(list_concentraciones[indicador]))
    
    plt.errorbar(data_integral[indicador]['Pressures'], 
             data_integral[indicador]['integrals'], 
             xerr= err_pressure,
             yerr = data_integral[indicador]['Err_int'],
             label=f'{100-i}/{i}', 
             fmt=fmt_list[indicador], markersize = 10,
             color=color)
    
    indicador += 1

ax.set_xlabel(r'Pressure (bar)', fontsize=15)
ax.set_ylabel(r'$\gamma$ / e$^-$ $\times$ $\lambda$', fontsize=15)
ax.tick_params(axis='both', which='major', labelsize=13)
ax.grid()
ax.legend(fontsize=13, title='Concentrations', title_fontsize=12)

sm = cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('Relative Pressure of N2', fontsize=13)
cbar.ax.tick_params(labelsize=12)

plt.tight_layout()

plt.savefig(f'Ar{Element}_Ultraviolet_phe.jpg', format='jpg', 
                bbox_inches='tight', dpi = 300) 

plt.show(block=False)
plt.pause(0.1)
input("Press enter to close all figures...")
plt.close('all')

