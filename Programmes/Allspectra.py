#%%

import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import numpy as np
from glob import glob
from Functions import extraer_presion, Sep_rut, Excel_value, RP, integral
import pandas as pd

err_SC = 0.1e-7 #A
err_SV = 50 #v
err_lambda = 5 #nm
err_pressure = 10e-3 #bar

element_mix = 'N2' 
Mix_Integrals = {}
list_concentraciones = [0.1, 0.5,1,5, 10]
integration_limits = [200, 300]

excel_path = r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\Whole_Data.xlsx"
base_path= rf'C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\{element_mix}'

data_sets = []
pressures_sets=[]
currents_sets=[]
voltages_sets=[]
titles = []
data_integral=[]

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

    title = f'Ar/N2 {100-Concentracion_N2}/{Concentracion_N2}'

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

    
    data_sets.append(data)
    pressures_sets.append(pressures)
    currents_sets.append(currents)
    voltages_sets.append(voltages)
    titles.append(title)

    print('Lista de presiones: ', pressures)
    integrales = []
    err_integral = []
    for ValPressureIndex in range(len(pressures)):
        sum, err = integral(data[ValPressureIndex], 
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

    data_integral.append(df)



Ruta_Candela = r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\CF4\5\5_bar\40kV40mA\After_WindowChange"

Candela_FD = RP(Ruta_Candela)
df_candela = Candela_FD['calibratedResults']

filters_candela = {
    'Element A': 'Ar',
    'Concentration A':  95,
    'Element B': 'CF4',
    'Concentration B': 5,
    'Pressure (bar)': 5
}

Current_candela = Excel_value(excel_path, filters_candela, 'SC')

NumElectronsCandela = Current_candela / (-1.602176634e-19)
err_NumE_candela = err_SC/ (-1.602176634e-19)

Candela_phe = df_candela['Counts'] / NumElectronsCandela
err_Candela_phe = np.sqrt((df_candela['Err_Counts']/NumElectronsCandela)**2+
                  (df_candela['Counts']*err_NumE_candela/(NumElectronsCandela)**2)**2)


print(f"Se cargaron {len(data_sets)} conjuntos de datos.")

#%%

# Compartir ejes entre los subplots (eje x por columnas, eje y entre primeras 2 columnas)
fig, axs = plt.subplots(2, 3, figsize=(16, 9))
# Compartir eje Y entre [0,0], [0,1], [1,0], [1,1]
axs[0, 1].sharey(axs[0, 0])
axs[1, 0].sharey(axs[0, 0])
axs[1, 1].sharey(axs[0, 0])

# Compartir eje X entre las columnas 0 y 1 solamente (en cada fila)
axs[1, 0].sharex(axs[0, 0])
axs[1, 1].sharex(axs[0, 1])

colormap = cm.turbo

for idx in range(6):
    row = idx // 3
    col = idx % 3
    ax = axs[row, col]

    if idx < 5:
        # Espectros
        data_i = data_sets[idx]
        pressures = pressures_sets[idx]
        title = titles[idx]

        norm = Normalize(vmin=min(pressures), vmax=max(pressures))

        for i in range(len(pressures)):
            color = colormap(norm(pressures[i]))
            ax.plot(data_i[i]['Lambda'], data_i[i]['Phe'],
                    color=color,
                    linewidth=0.5)
            ax.fill_between(data_i[i]['Lambda'],
                            data_i[i]['Phe'] - data_i[i]['Err_Phe'],
                            data_i[i]['Phe'] + data_i[i]['Err_Phe'],
                            color=color,
                            alpha=0.5)

        ax.fill_between(df_candela['Lambda'],
                        Candela_phe - err_Candela_phe,
                        Candela_phe + err_Candela_phe,
                        label=f'Ar/CF4 95/5 5bar {Current_candela}A',
                        color='magenta',
                        alpha=0.5)

        ax.set_title(title, fontsize=11)
        ax.tick_params(axis='both', which='major', labelsize=9, labelbottom=True)
        ax.grid()
        ax.legend(fontsize=8, title_fontsize=8)
        ax.set_xlim(integration_limits[0], integration_limits[1])
        ax.set_ylim(-1,20)

        if row == 1:
            ax.set_xlabel(r'$\lambda$ (nm)', fontsize=10)
        if col == 0:
            ax.set_ylabel(r'Photons / electrons (A.U)', fontsize=10)

    # elif idx == 4:
    #     # Plot vacío visible con ejes compartidos
    #     ax.set_title('Sin datos (vacío)', fontsize=11)
    #     ax.set_xlabel(r'$\lambda$ (nm)', fontsize=10)
    #     if col == 0:
    #         ax.set_ylabel(r'Photons / electrons (A.U)', fontsize=10)
    #     ax.tick_params(axis='both', which='major', labelsize=9)
    #     ax.grid()

    elif idx == 5:
        # Plot de integrales
        ax.set_title('Integrals vs Pressure', fontsize=11)

        for idx_int, df in enumerate(data_integral):
            color = colormap(idx_int / len(data_integral))
            ax.errorbar(df['Pressures'], df['integrals'], yerr=df['Err_int'], xerr = err_pressure,
                        fmt='o-', label=f'{100 - list_concentraciones[idx_int]}/{list_concentraciones[idx_int]}',
                        color=color)

        ax.set_xlabel('Pressure (bar)', fontsize=10)
        ax.set_ylabel(r'$\gamma$/e$^-$ $\times \lambda$', fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=9)
        ax.grid()
        ax.legend(fontsize=8, title='Ar/N2', title_fontsize=8)

fig.subplots_adjust(right=0.89, left=0.04, hspace=0.2, wspace=0.18)
cbar_ax = fig.add_axes([0.9, 0.15, 0.015, 0.7])
sm = cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])
fig.colorbar(sm, cax=cbar_ax, label='Pressure (bar)')

plt.savefig('AllSpectra.jpg', format='jpg',
            bbox_inches='tight', dpi=300)


#%%

# Crear figura con 1 fila y 2 columnas
fig, (ax_sc, ax_sv) = plt.subplots(1, 2, figsize=(16, 9), sharex=True)

# Colores por mezcla
colors = plt.cm.turbo(np.linspace(0, 1, len(currents_sets)))

# === Plot SC vs Pressure ===
for i, (pressures, currents, title) in enumerate(zip(pressures_sets, currents_sets, titles)):
    ax_sc.errorbar(pressures, currents,
                   yerr = err_SC, xerr = err_pressure, 
                   fmt = 'o-', color=colors[i], label=title)

ax_sc.set_xlabel('Pressure (bar)', fontsize=14)
ax_sc.set_ylabel('Saturation Current (A)', fontsize=14)
ax_sc.set_title('SC vs Pressure', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax_sc.grid(True)
ax_sc.legend(fontsize=14)

# === Plot SV vs Pressure ===
for i, (pressures, voltages, title) in enumerate(zip(pressures_sets, voltages_sets, titles)):
    ax_sv.errorbar(pressures, voltages,
                   yerr = err_SV, xerr = err_pressure,
                   fmt='s-', color=colors[i], label=title)

ax_sv.set_xlabel('Pressure (bar)', fontsize=14)
ax_sv.set_ylabel('Saturation Voltage (V)', fontsize=14)
ax_sv.set_title('SV vs Pressure', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax_sv.grid(True)
ax_sv.legend(fontsize=9)

# Ajustar y mostrar
plt.tight_layout()
plt.savefig('SC_SV_vs_Pressure.jpg', dpi=300, bbox_inches='tight')
plt.show(block=False)
plt.pause(0.1)
input("Presiona ENTER para cerrar la figura...")
plt.close('all')