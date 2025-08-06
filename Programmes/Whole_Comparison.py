#%%
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import pandas as pd
import numpy as np
import math as math
from glob import glob
from scipy.stats import norm
from Functions import Sep_rut, Excel_value, extraer_presion, integral


def Data_Load(rutas_ordenadas, intlimits):

    pressures = []
    currents = []
    voltages = []
    data = []

    Element, Concentracion = Sep_rut(rutas_ordenadas[0])[0:2]

    for i in rutas_ordenadas:

        Presion, Volt_Amp, Ar_Concentration = Sep_rut(i)[2:6]

        filters = {
            'Element A': 'Ar',
            'Concentration A':  Ar_Concentration,
            'Element B': Element,
            'Concentration B': Concentracion,
            'Pressure (bar)': Presion
        }

        Saturation_current = Excel_value(filters, 'SC')
        Saturation_volt = Excel_value(filters, 'SV')

        if Excel_value(filters, 'C3kV')==0:
            print('Current at 3kV do not exist. Using standard error of', err_SC_standard)
            err_SC = err_SC_standard

        else:
            Saturation_current_3kV = Excel_value(filters, 'C3kV')
            err_SC  = abs(Saturation_current_3kV - Saturation_current)
            print('Current:', Saturation_current)
            print('Current at 3kV:', Saturation_current_3kV)
            print('Current Error: ', err_SC)

        pattern_data = os.path.join(i, 'Analized_Data', '*_AllData.txt')
        archivos = glob(pattern_data)
        
        if archivos:
            archivo = archivos[0]
            df = pd.read_csv(archivo, sep='\t', engine='python')
        else:
            print(f"No se encontró archivo en {i}")

        pressures.append(Presion)
        currents.append(Saturation_current)
        voltages.append(Saturation_volt)
        data.append(df)

        # [0]: Corriente de Saturacion
        # [1]: Voltaje de Saturacion
        # [2]: Numero de Electrones
        # [3]: Array photons per electron

    print('Lista de presiones: ', pressures)
    integrales = []
    err_integral = []
    for ValPressureIndex in range(len(pressures)):
        sum, err = integral(data[ValPressureIndex].clip(lower=0), 
                        'Lambda', err_lambda,
                        'Phe', 'Err_Phe',
                        *intlimits)
        integrales.append(sum)
        err_integral.append(err)

    df_integrals = pd.DataFrame({
            'Pressures': pressures,
            'integrals': integrales, 
            'Err_int': err_integral
            })

    return pressures, currents, voltages, data, df_integrals, Element, Concentracion

# ------ Errors -----------

err_SC_standard = 0.05e-7
err_SV = 50 #v
err_lambda = 5 #nm
err_pressure = 10e-3 #bar

# ----------- Integration Limits -----------------

integration_limits = [300, 450]
integration_limits_ref = [500, 750]

# -------- element and concentration -------------

element_mix = 'N2'
Lista_Concentraciones = [5,20,50]
folders_to_compare = ['0V', '0V_withNO']

# ----------------- DATA ------------------------

pressures_sets = []
data_sets = []
integral_sets = []
titles_sets = []

for Concentracion_N2 in Lista_Concentraciones:

    if '.' in str(Concentracion_N2):
        Concentration_mix = str(Concentracion_N2).replace('.','-')

    else:
        Concentration_mix = Concentracion_N2

    # -------- Paths ------------

    base_path = rf'C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\{element_mix}\{Concentration_mix}'
    pattern = os.path.join(base_path, '*_bar', '40kV40mA', f'{folders_to_compare[0]}')

    rutas = glob(pattern)
    rutas_ordenadas = sorted(rutas, key=extraer_presion)

    base_path_MC = rf'C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\{element_mix}\{Concentration_mix}'
    pattern_MC = os.path.join(base_path_MC, '*_bar', '40kV40mA', f'{folders_to_compare[1]}')

    rutas_MC = glob(pattern_MC)
    rutas_ordenadas_MC = sorted(rutas_MC, key=extraer_presion)

    pressures, currents, voltages, data, df_integrals, Element, Concentracion = Data_Load(rutas_ordenadas, integration_limits)
    pressures_MC, currents_MC, voltages_MC, data_MC, df_integrals_MC, Element_MC, Concentracion_MC = Data_Load(rutas_ordenadas_MC, integration_limits)

    if pressures == pressures_MC:
        print('Pressures of both paths are the same')

    else:
        print('There is a discerpancy in the pressures of each path.')
        sys.exit()

    title = f'Ar/{Element} {100 - Concentracion}/{Concentracion}'

    title_rest = f'{title}: Spectrums substraction'
    title_coeficient = f'Scintillation yield coeficient'

    pressures_sets.append([pressures, pressures_MC])
    data_sets.append([data, data_MC])
    integral_sets.append([df_integrals, df_integrals_MC])
    titles_sets.append([title_rest, title_coeficient])

print(integral_sets[0][1])

#%%
if len(Lista_Concentraciones) > 1:

    fig, axs = plt.subplots(3, 2, figsize=(8, 12), sharex='col', sharey='col')

    colormap = cm.turbo


    for idx in range(3):
        row = idx
        ax_spectrum = axs[row, 0]
        ax_integral = axs[row, 1]

        data_i = data_sets[idx]
        integral_data_i = integral_sets[idx]
        title_i = titles_sets[idx]
        pressures = pressures_sets[idx][0]

        norm = Normalize(vmin=min(pressures), vmax=max(pressures))

        # --------- Columna izquierda: diferencia de espectros --------
        for i in range(len(pressures)):
            color = colormap(norm(pressures[i]))
            diff = data_i[1][i]['Phe'] - data_i[0][i]['Phe'] #Substraction folder_to_compare[1] - folder_to_compare[0]
            err_diff = np.sqrt(data_i[0][i]['Err_Phe']**2 + data_i[1][i]['Err_Phe']**2)

            ax_spectrum.plot(data_i[0][i]['Lambda'], diff, label=f'{pressures[i]} bar',
                            color=color, linewidth=0.5)
            ax_spectrum.fill_between(data_i[0][i]['Lambda'], diff - err_diff, diff + err_diff,
                                    color=color, alpha=0.5)
            
            ax_spectrum.set_title(f'{title_i[0]}')

        # --------- Columna derecha: diferencia de integrales --------
        coef_diff = integral_data_i[1]['integrals']/integral_data_i[0]['integrals']
        err_diff = np.sqrt((integral_data_i[1]['Err_int']/integral_data_i[0]['integrals'])**2 + 
                        (integral_data_i[1]['integrals']*integral_data_i[0]['Err_int']/integral_data_i[0]['integrals'])**2)

        ax_integral.errorbar(integral_data_i[0]['Pressures'], coef_diff,
                            yerr = err_diff, label='Coefficientsintegral', 
                            color='black', fmt='o:')
        ax_integral.fill_between(integral_data_i[0]['Pressures'],
                                coef_diff - err_diff,
                                coef_diff + err_diff,
                                color='lightblue', alpha=0.3)
        
        ax_integral.set_title(f'{title_i[1]}')

        # --------- Ejes, etiquetas --------
        for ax in [ax_spectrum, ax_integral]:
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.grid(True)
            ax.legend(fontsize=8)

        if row == 2:
            ax_spectrum.set_xlabel(r'Wavelength (nm)', fontsize=10)
            ax_integral.set_xlabel(r'Pressure (bar)', fontsize=10)

        ax_spectrum.set_ylabel(r'Photons / electrons (A.U)', fontsize=10)
        ax_integral.set_ylabel(r'Relative Scintillation Yield', fontsize=10)


    plt.tight_layout()
    plt.savefig(f'ArN2_MCcomparison.jpg',
                format='jpg', bbox_inches='tight', dpi=300)
    plt.show(block=False)
    plt.pause(0.1)
    input("Press enter to close all figures...")
    plt.close('all')

else: 

    data = data_sets[0][0]
    data_MC = data_sets[0][1]
    title_i = titles_sets[0][0]
    pressures = pressures_sets[0][0]

    norm = Normalize(vmin=min(pressures), vmax=max(pressures))
    colormap = plt.colormaps['turbo']

    fig, ax1 = plt.subplots(figsize=(16, 9))

    # --- Primer gráfico: espectros ---
    for i in range(len(pressures)):

        diff = data_MC[i]['Phe'] - data[i]['Phe']
        err_diff = np.sqrt(data_MC[i]['Err_Phe']**2 + data[i]['Err_Phe']**2)

        color = colormap(norm(pressures[i]))
        ax1.plot(data[i]['Lambda'], diff, color=color, linewidth=0.5)

        ax1.fill_between(data[i]['Lambda'],
                        diff - err_diff,
                        diff + err_diff,
                        label=f'{pressures[i]} bar',
                        color=color, alpha=0.5)

    ax1.set_xlabel(r'wavelenght (nm)', fontsize=15)
    ax1.set_ylabel(r'Photons / electrons (A.U)', fontsize=15)
    ax1.tick_params(axis='both', which='major', labelsize=13)
    ax1.grid()
    ax1.legend(fontsize=10, title='Pressures', title_fontsize=11)
    ax1.set_title(f'Ar/{Element} {100-Concentracion}/{Concentracion}', fontsize=14)

    plt.tight_layout()
    plt.savefig(f'Ar{Element}_{100-Concentracion}{Concentracion}_MCcomparison.jpg',
                format='jpg', bbox_inches='tight', dpi=300)

    plt.show(block=False)
    plt.pause(0.1)
    input("Press enter to close all figures...")
    plt.close('all')
